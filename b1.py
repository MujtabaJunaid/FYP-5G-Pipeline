# base_station.py (FIXED) - Real spectrum monitoring and energy-based protection
# - Actual channel monitoring instead of time-based protection
# - Real-time spectrum occupancy tracking
# - Statistical energy detection for primary user protection

import socket
import threading
import json
import struct
import time
import os
import math
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import hashlib
from reedsolo import RSCodec
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import numpy as np
import scipy.special
import random

# --- CONFIGURATION ---
BS_INSTANCE = 1
# ---------------------

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

HOST = "127.0.0.1"
COMM_RANGE = 150.0

if BS_INSTANCE == 1:
    PORT = 50050
    BS_ID = "BS1"
    BS_POS = (0.0, 0.0)
    NEIGHBORS = [("127.0.0.1", 50051)]
    PDF_PATH = os.path.join(RESULTS_DIR, "base_station_1_report.pdf")
else:
    PORT = 50051
    BS_ID = "BS2"
    BS_POS = (200.0, 0.0)
    NEIGHBORS = [("127.0.0.1", 50050)]
    PDF_PATH = os.path.join(RESULTS_DIR, "base_station_2_report.pdf")

def cleanup_old_files():
    """Delete old base station report files"""
    pattern = os.path.join(RESULTS_DIR, f"base_station_{BS_INSTANCE}_*")
    old_files = glob.glob(pattern)
    for file_path in old_files:
        try:
            os.remove(file_path)
            print(f"[BS {BS_ID}] Removed old file: {file_path}")
        except Exception as e:
            print(f"[BS {BS_ID}] Error removing {file_path}: {e}")

# Clean up old files on startup
cleanup_old_files()

clients_lock = threading.Lock()
clients = {}
retry_queue = deque()
retry_queue_lock = threading.Lock()
stats = { "received": 0, "forwarded_local": 0, "forwarded_remote": 0, "queued": 0, "delivered": 0, "per_hop_counts": [] }
stats_lock = threading.Lock()
STOP = threading.Event()

# ---------- SPECTRUM MONITORING CONFIG ----------
SPECTRUM_MONITOR_WINDOW = 1000  # Samples for spectrum monitoring
MONITOR_UPDATE_INTERVAL = 2.0   # Update spectrum state every 2 seconds
FALSE_ALARM_PROB = 0.05         # 5% false alarm probability for BS monitoring
NOISE_VARIANCE_ESTIMATE = 0.01  # Initial noise variance estimate

# ---------- Crypto & RS ----------
PASSPHRASE = b"very secret passphrase - change this!"
KEY = hashlib.sha256(PASSPHRASE).digest()
rs = RSCodec(40)

# ---------- Primary list ----------
PRIMARY_SENDERS = ["JAZZ", "UFONE", "TELENOR", "WARID", "STARLINK", "ZONG", "SCO", "PTCL"]

# ---------- REAL SPECTRUM MONITORING ----------
class SpectrumMonitor:
    def __init__(self):
        self.spectrum_state = "IDLE"  # "IDLE" or "BUSY"
        self.energy_history = []
        self.last_update = 0
        self.detection_threshold = 0
        self.noise_variance = NOISE_VARIANCE_ESTIMATE
        self.lock = threading.Lock()
        
    def generate_channel_samples(self):
        """Simulate receiving actual channel samples at base station"""
        # Simulate realistic channel conditions with occasional primary users
        time_now = time.time()
        
        # Simulate primary user activity patterns (more realistic than random)
        primary_active = False
        if int(time_now) % 20 < 6:  # Primary active 30% of the time in 20s cycles
            primary_active = True
            
        if primary_active:
            # Primary user signal (stronger than secondary)
            primary_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], SPECTRUM_MONITOR_WINDOW)
            primary_signal = primary_symbols * 1.2  # Strong primary signal
            noise = np.sqrt(self.noise_variance/2) * (np.random.randn(SPECTRUM_MONITOR_WINDOW) + 
                                                    1j*np.random.randn(SPECTRUM_MONITOR_WINDOW))
            received_signal = primary_signal + noise
        else:
            # Background noise with occasional weak secondary signals
            if random.random() < 0.4:  # 40% chance of weak secondary activity
                secondary_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], SPECTRUM_MONITOR_WINDOW//2)
                secondary_signal = secondary_symbols * 0.3  # Weak secondary signal
                padding = np.zeros(SPECTRUM_MONITOR_WINDOW - len(secondary_signal), dtype=complex)
                received_signal = np.concatenate([secondary_signal, padding])
                noise = np.sqrt(self.noise_variance/2) * (np.random.randn(SPECTRUM_MONITOR_WINDOW) + 
                                                        1j*np.random.randn(SPECTRUM_MONITOR_WINDOW))
                received_signal += noise
            else:
                # Noise only
                received_signal = np.sqrt(self.noise_variance/2) * (
                    np.random.randn(SPECTRUM_MONITOR_WINDOW) + 1j*np.random.randn(SPECTRUM_MONITOR_WINDOW))
        
        return received_signal, primary_active
    
    def calculate_detection_threshold(self, p_fa=FALSE_ALARM_PROB):
        """Calculate energy detection threshold using Neyman-Pearson criterion"""
        N = SPECTRUM_MONITOR_WINDOW
        Q_inv = np.sqrt(2) * scipy.special.erfinv(1 - 2 * p_fa)
        threshold = self.noise_variance * (1 + Q_inv / np.sqrt(N/2))
        return threshold
    
    def update_spectrum_state(self):
        """Perform spectrum sensing and update channel state"""
        with self.lock:
            # Get channel samples
            channel_samples, actual_primary = self.generate_channel_samples()
            
            # Calculate test statistic (energy)
            test_statistic = np.sum(np.abs(channel_samples)**2) / len(channel_samples)
            
            # Update noise variance estimate (sliding average)
            self.noise_variance = 0.95 * self.noise_variance + 0.05 * np.var(channel_samples)
            
            # Calculate detection threshold
            self.detection_threshold = self.calculate_detection_threshold()
            
            # Make decision
            self.spectrum_state = "BUSY" if test_statistic > self.detection_threshold else "IDLE"
            
            # Store energy measurement
            self.energy_history.append({
                'timestamp': time.time(),
                'test_statistic': test_statistic,
                'threshold': self.detection_threshold,
                'state': self.spectrum_state,
                'actual_primary': actual_primary,
                'snr': 10 * np.log10(test_statistic / self.noise_variance) if self.noise_variance > 0 else -100
            })
            
            # Keep only recent history (last 100 measurements)
            if len(self.energy_history) > 100:
                self.energy_history = self.energy_history[-100:]
            
            self.last_update = time.time()
            
            return self.spectrum_state, test_statistic, self.detection_threshold
    
    def get_spectrum_state(self):
        """Get current spectrum state with automatic update if stale"""
        with self.lock:
            if time.time() - self.last_update > MONITOR_UPDATE_INTERVAL:
                self.update_spectrum_state()
            return self.spectrum_state
    
    def can_secondary_transmit(self, sender_id):
        """Check if secondary user can transmit based on actual spectrum conditions"""
        if sender_id.upper() in PRIMARY_SENDERS:
            return True  # Primary users always allowed
        
        current_state = self.get_spectrum_state()
        return current_state == "IDLE"

# Initialize spectrum monitor
spectrum_monitor = SpectrumMonitor()

def spectrum_monitoring_worker():
    """Background thread for continuous spectrum monitoring"""
    while not STOP.is_set():
        try:
            state, test_stat, threshold = spectrum_monitor.update_spectrum_state()
            if state == "BUSY":
                print(f"[BS {BS_ID}] SPECTRUM MONITOR: Channel BUSY (Energy: {test_stat:.4e}, Threshold: {threshold:.4e})")
            # else:
            #     print(f"[BS {BS_ID}] SPECTRUM MONITOR: Channel IDLE (Energy: {test_stat:.4e}, Threshold: {threshold:.4e})")
            
            time.sleep(MONITOR_UPDATE_INTERVAL)
        except Exception as e:
            print(f"[BS {BS_ID}] Spectrum monitoring error: {e}")
            time.sleep(1)

# ---------- Helper Functions ----------
def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def recv_full(sock, length):
    data = b""
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more: raise ConnectionError("socket closed")
        data += more
    return data

def send_all_with_retry(conn, data):
    try:
        conn.sendall(data)
        return True
    except Exception as e:
        try:
            peer = conn.getpeername()
        except Exception:
            peer = None
        print(f"[BS {BS_ID}] Forward/send failed to {peer}: {e}")
        return False

# ---------- PHY helpers for energy computation ----------
def bits_from_bytes(b: bytes):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

def qam_mod(bits, M):
    k = int(np.log2(M))
    if len(bits) % k != 0:
        bits = np.concatenate([bits, np.zeros(k - (len(bits) % k), dtype=np.uint8)])
    ints = bits.reshape((-1, k)).dot(1 << np.arange(k - 1, -1, -1))
    sqrtM = int(np.sqrt(M))
    x_int = ints % sqrtM
    y_int = ints // sqrtM
    raw_x = 2 * x_int - (sqrtM - 1)
    raw_y = 2 * y_int - (sqrtM - 1)
    scale = np.sqrt((2.0 / 3.0) * (M - 1)) if M > 1 else 1.0
    return (raw_x / scale) + 1j * (raw_y / scale)

def ofdm_mod(symbols, num_subcarriers, cp_len):
    if len(symbols) == 0:
        return np.array([])
    n_ofdm = int(np.ceil(len(symbols) / num_subcarriers))
    padded = np.pad(symbols, (0, n_ofdm * num_subcarriers - len(symbols)))
    reshaped = padded.reshape((n_ofdm, num_subcarriers))
    ifft_data = np.fft.ifft(reshaped, axis=1)
    ofdm_with_cp = np.hstack([ifft_data[:, -cp_len:], ifft_data])
    return ofdm_with_cp.flatten()

def compute_energy_from_plaintext(plaintext):
    """Given plaintext as struct(M,nc,cp)+pkt, reconstruct bits->OFDM and compute average energy"""
    try:
        M, nc, cp = struct.unpack(">H H B", plaintext[:5])
        pkt = plaintext[5:]
        msg_len = struct.unpack(">H", pkt[:2])[0]
        msg_bytes = pkt[2:2+msg_len]
        bits = bits_from_bytes(msg_bytes)
        tx_symbols = qam_mod(bits, M)
        ofdm_sig = ofdm_mod(tx_symbols, nc, cp)
        energy = float(np.mean(np.abs(ofdm_sig)**2)) if ofdm_sig.size>0 else 0.0
        return energy, ofdm_sig, M, nc, cp, msg_bytes.decode("utf-8", errors="replace")
    except Exception:
        return 0.0, np.array([]), 16, 64, 8, "<decode_error>"

def process_incoming_frame(src_id, dst_id, raw_payload_after_dst, hop_count=0):
    """
    Process a frame incoming from a local client with REAL spectrum monitoring.
    """
    # Immediately log raw receipt so we always have a BS-side record
    try:
        print(f"[BS {BS_ID}] Raw frame received: src={src_id} dst={dst_id} len={len(raw_payload_after_dst)} bytes")
    except Exception:
        print(f"[BS {BS_ID}] Raw frame received (src/dst len logging failed)")

    encoded_bytes_with_possible_header = raw_payload_after_dst

    energy = 0.0
    ofdm_sig = np.array([])
    M = 16; nc = 64; cp = 8; msg_text = "<unknown>"
    sensing_header_present = False
    parsed_plaintext = None

    # Try Path A: if there are at least 8 bytes, interpret first 8 as sensing header, then RS-decode remainder
    if len(encoded_bytes_with_possible_header) >= 8:
        possible_hdr = encoded_bytes_with_possible_header[:8]
        remainder = encoded_bytes_with_possible_header[8:]
        try:
            sensing_energy_tmp = struct.unpack(">d", possible_hdr)[0]
            rs_decoded = rs.decode(remainder)[0]
            plaintext = aes_gcm_decrypt(rs_decoded, KEY)
            sensing_header_present = True
            parsed_plaintext = plaintext
            energy = float(sensing_energy_tmp)
            try:
                _energy, _ofdm_sig, M, nc, cp, msg_text = compute_energy_from_plaintext(plaintext)
                if _ofdm_sig.size > 0:
                    ofdm_sig = _ofdm_sig
                    energy = _energy
            except Exception:
                pass
        except Exception:
            parsed_plaintext = None
            sensing_header_present = False

    # Path B: if Path A failed or not attempted, try RS-decode full buffer
    if parsed_plaintext is None:
        try:
            rs_decoded = rs.decode(encoded_bytes_with_possible_header)[0]
            plaintext = aes_gcm_decrypt(rs_decoded, KEY)
            parsed_plaintext = plaintext
            energy, ofdm_sig, M, nc, cp, msg_text = compute_energy_from_plaintext(plaintext)
        except Exception:
            parsed_plaintext = None

    # REAL SPECTRUM ACCESS CONTROL (replacing time-based protection)
    is_primary_sender = (src_id.upper() in PRIMARY_SENDERS)
    
    if not is_primary_sender:
        # Check if secondary user can transmit based on actual spectrum conditions
        can_transmit = spectrum_monitor.can_secondary_transmit(src_id)
        if not can_transmit:
            current_state = spectrum_monitor.get_spectrum_state()
            print(f"[BS {BS_ID}] BLOCKING secondary transmission from {src_id} - spectrum {current_state}")
            with stats_lock:
                stats["queued"] += 1
            item = {"src_id": src_id, "dst_id": dst_id, "payload": encoded_bytes_with_possible_header, 
                   "hop_count": hop_count, "ts": time.time()}
            with retry_queue_lock:
                retry_queue.append(item)
            return

    # Record energy measurement for monitoring
    try:
        spectrum_monitor.energy_history.append({
            'timestamp': time.time(),
            'test_statistic': float(energy),
            'threshold': spectrum_monitor.detection_threshold,
            'state': "TX_ACTIVE",
            'actual_primary': is_primary_sender,
            'snr': 10 * np.log10(energy / spectrum_monitor.noise_variance) if spectrum_monitor.noise_variance > 0 else -100,
            'sender': src_id
        })
    except Exception:
        pass

    # Attempt immediate delivery (forward EXACT payload as received after dst id, preserving sensing header if any)
    with clients_lock:
        dst_info = clients.get(dst_id)
    if dst_info and distance(BS_POS, dst_info.get("pos", (0,0))) <= COMM_RANGE:
        src_bytes = src_id.encode("utf-8")
        payload = struct.pack(">H", len(src_bytes)) + src_bytes + encoded_bytes_with_possible_header
        wire_frame = struct.pack(">I", len(payload)) + payload

        if send_all_with_retry(dst_info["conn"], wire_frame):
            print(f"[BS {BS_ID}] Delivered from {src_id} -> local {dst_id} (Primary: {is_primary_sender})")
            with stats_lock:
                stats["forwarded_local"] += 1; stats["delivered"] += 1
                stats["per_hop_counts"].append(hop_count)
            return
        else:
            print(f"[BS {BS_ID}] Failed to deliver locally to {dst_id}, will attempt hop/queue.")

    # Not local or couldn't deliver: hop to neighbor(s)
    src_bytes, dst_bytes = src_id.encode("utf-8"), dst_id.encode("utf-8")
    hop_payload = (struct.pack(">B", hop_count + 1) +
                   struct.pack(">H", len(src_bytes)) + src_bytes +
                   struct.pack(">H", len(dst_bytes)) + dst_bytes + encoded_bytes_with_possible_header)
    hop_wire = struct.pack(">I", len(hop_payload)) + hop_payload
    
    forwarded = False
    for nb_host, nb_port in NEIGHBORS:
        try:
            with socket.create_connection((nb_host, nb_port), timeout=2.0) as nb_sock:
                nb_sock.sendall(hop_wire)
                forwarded = True
                print(f"[BS {BS_ID}] Hopped message from {src_id} -> neighbor for {dst_id}")
                with stats_lock: stats["forwarded_remote"] += 1
                break
        except Exception as e:
            # neighbor hop failed: continue to next neighbor
            continue

    if not forwarded:
        print(f"[BS {BS_ID}] Queued message for {dst_id} (no neighbor available)")
        with stats_lock: stats["queued"] += 1
        item = {"src_id": src_id, "dst_id": dst_id, "payload": encoded_bytes_with_possible_header, "hop_count": hop_count, "ts": time.time()}
        with retry_queue_lock: retry_queue.append(item)

def handle_registered_client(conn, addr):
    client_id = None
    try:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(1024)
            if not chunk: raise ConnectionError("Client closed before registration")
            buf += chunk
        
        line, buffer = buf.split(b"\n", 1)
        reg = json.loads(line.decode("utf-8"))
        client_id = reg["id"]
        with clients_lock:
            clients[client_id] = {"conn": conn, "addr": addr, "type": reg["type"], "pos": tuple(reg["pos"])}
        print(f"[BS {BS_ID}] Registered {reg['type']} '{client_id}' at {reg['pos']} from {addr}")

        while not STOP.is_set():
            while len(buffer) < 4:
                more = conn.recv(4096)
                if not more: raise ConnectionError("Client closed")
                buffer += more
            length, buffer = struct.unpack(">I", buffer[:4])[0], buffer[4:]
            
            while len(buffer) < length:
                more = conn.recv(4096)
                if not more: raise ConnectionError("Client closed during payload")
                buffer += more
            payload, buffer = buffer[:length], buffer[length:]

            dst_len = struct.unpack(">H", payload[:2])[0]
            dst_id = payload[2:2+dst_len].decode("utf-8")
            remaining = payload[2+dst_len:]  # KEEP the original bytes after the dst id (may include sensing header)

            # Immediate logging of frame receipt (guarantees BS records every frame)
            try:
                print(f"[BS {BS_ID}] Received frame from {client_id} -> {dst_id} ({len(remaining)} bytes after dst id)")
            except Exception:
                pass

            with stats_lock:
                stats["received"] += 1

            # Pass the entire remaining bytes (including header if present) to process_incoming_frame
            threading.Thread(target=process_incoming_frame, args=(client_id, dst_id, remaining), daemon=True).start()

    except (ConnectionError, ConnectionResetError, struct.error, json.JSONDecodeError):
        pass
    finally:
        if client_id:
            with clients_lock: clients.pop(client_id, None)
            print(f"[BS {BS_ID}] Client {client_id} disconnected.")
        conn.close()

def handle_hop_connection(conn, addr):
    try:
        payload = recv_full(conn, struct.unpack(">I", recv_full(conn, 4))[0])
        hop_count = payload[0]
        pos = 1
        src_len = struct.unpack(">H", payload[pos:pos+2])[0]; pos += 2
        src_id = payload[pos:pos+src_len].decode("utf-8"); pos += src_len
        dst_len = struct.unpack(">H", payload[pos:pos+2])[0]; pos += 2
        dst_id = payload[pos:pos+dst_len].decode("utf-8"); pos += dst_len
        encoded_bytes = payload[pos:]
        
        print(f"[BS {BS_ID}] Received hop from {addr} for {src_id} -> {dst_id}")
        with stats_lock: stats["received"] += 1
        process_incoming_frame(src_id, dst_id, encoded_bytes, hop_count)
    except Exception as e:
        print(f"[BS {BS_ID}] Error handling hop from {addr}: {e}")
    finally:
        conn.close()

def accept_loop():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, PORT))
        sock.listen(16)
        print(f"[BS {BS_ID}] Listening on {HOST}:{PORT} (pos={BS_POS}, range={COMM_RANGE})")
        print(f"[BS {BS_ID}] Spectrum monitoring ACTIVE - Primary protection: {PRIMARY_SENDERS}")
        while not STOP.is_set():
            try:
                conn, addr = sock.accept()
                first_char = conn.recv(1, socket.MSG_PEEK)
                if first_char == b'{':
                    threading.Thread(target=handle_registered_client, args=(conn, addr), daemon=True).start()
                else:
                    threading.Thread(target=handle_hop_connection, args=(conn, addr), daemon=True).start()
            except Exception:
                continue

# ---------- Retry thread with spectrum awareness ----------
def retry_worker():
    while not STOP.is_set():
        try:
            with retry_queue_lock:
                if not retry_queue:
                    pass
                else:
                    item = retry_queue.popleft()
                    src_id = item["src_id"]
                    
                    # Check if this is a secondary user and spectrum is available
                    if src_id.upper() not in PRIMARY_SENDERS:
                        can_transmit = spectrum_monitor.can_secondary_transmit(src_id)
                        if not can_transmit:
                            # Spectrum still busy, put back in queue
                            retry_queue.append(item)
                            time.sleep(0.5)
                            continue
                    
                    # Spectrum available or primary user - attempt delivery
                    process_incoming_frame(item["src_id"], item["dst_id"], item["payload"], item.get("hop_count", 0))
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
            continue

# ---------- Enhanced Reporting ----------
def make_spectrum_monitoring_plot(monitor):
    """Create comprehensive spectrum monitoring visualization"""
    if not monitor.energy_history:
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No spectrum monitoring data", ha='center', va='center')
        plt.title("Spectrum Monitoring History")
        return fig
    
    history = monitor.energy_history[-50:]  # Last 50 measurements
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Energy timeline
    timestamps = [h['timestamp'] - history[0]['timestamp'] for h in history]
    test_stats = [h['test_statistic'] for h in history]
    thresholds = [h['threshold'] for h in history]
    
    ax1.plot(timestamps, test_stats, 'b-', label='Test Statistic', linewidth=1)
    ax1.plot(timestamps, thresholds, 'r--', label='Detection Threshold', linewidth=1)
    ax1.fill_between(timestamps, test_stats, thresholds, where=np.array(test_stats) > np.array(thresholds), 
                    alpha=0.3, color='red', label='BUSY')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Energy')
    ax1.set_title('Spectrum Energy Timeline')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: State distribution
    states = [h['state'] for h in history]
    state_counts = {state: states.count(state) for state in set(states)}
    ax2.bar(state_counts.keys(), state_counts.values(), color=['green', 'red', 'blue'])
    ax2.set_title('Channel State Distribution')
    ax2.set_ylabel('Count')
    
    # Plot 3: SNR distribution
    snrs = [h.get('snr', -100) for h in history if h.get('snr', -100) > -50]
    if snrs:
        ax3.hist(snrs, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('SNR (dB)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('SNR Distribution')
        ax3.grid(True)
    
    # Plot 4: Primary vs Secondary activity
    primary_tx = sum(1 for h in history if h.get('actual_primary', False))
    secondary_tx = sum(1 for h in history if h.get('sender') and h.get('sender').upper() not in PRIMARY_SENDERS)
    ax4.bar(['Primary', 'Secondary'], [primary_tx, secondary_tx], color=['red', 'blue'])
    ax4.set_title('Transmission Activity')
    ax4.set_ylabel('Count')
    
    plt.tight_layout()
    return fig

def export_report():
    try:
        figs = []
        
        # Spectrum monitoring plot
        figs.append(make_spectrum_monitoring_plot(spectrum_monitor))
        
        # Stats summary figure
        fig2 = plt.figure(figsize=(8, 6))
        current_state = spectrum_monitor.get_spectrum_state()
        stats_text = f"""
        Base Station {BS_ID} - Performance Report
        
        Spectrum State: {current_state}
        Noise Variance: {spectrum_monitor.noise_variance:.4e}
        Detection Threshold: {spectrum_monitor.detection_threshold:.4e}
        
        Network Statistics:
        - Frames Received: {stats.get('received',0)}
        - Local Forwarded: {stats.get('forwarded_local',0)}
        - Remote Forwarded: {stats.get('forwarded_remote',0)}
        - Queued Frames: {stats.get('queued',0)}
        - Delivered Frames: {stats.get('delivered',0)}
        - Primary Events: {stats.get('primary_events',0)}
        
        Primary Users: {', '.join(PRIMARY_SENDERS)}
        """
        plt.text(0.02, 0.5, stats_text, fontsize=10, va='center', family='monospace')
        plt.axis('off')
        figs.append(fig2)

        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        pdf_path = os.path.join(RESULTS_DIR, f"base_station_{BS_INSTANCE}_report_{timestamp}.pdf")
        with PdfPages(pdf_path) as pdf:
            for f in figs:
                pdf.savefig(f)
                plt.close(f)
        print(f"[BS {BS_ID}] Exported comprehensive report to {pdf_path}")
    except Exception as e:
        print(f"[BS {BS_ID}] Error exporting report: {e}")

def main():
    # Start spectrum monitoring thread
    threading.Thread(target=spectrum_monitoring_worker, daemon=True).start()
    
    # Start retry worker
    threading.Thread(target=retry_worker, daemon=True).start()
    
    try:
        accept_loop()
    except KeyboardInterrupt:
        print(f"\n[BS {BS_ID}] Shutting down...")
    finally:
        STOP.set()
        # Export comprehensive report before exit
        export_report()

if __name__ == "__main__":
    main()