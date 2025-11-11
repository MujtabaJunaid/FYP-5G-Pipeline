# Dynamic_Receiver.py (UPDATED) - Spectrum sensing uses actual transmitted bits
# - Implements per-peer sensing using real message bits (no random bits)
# - Retains robust socket, RS, AES handling and plotting

import socket, struct, threading, json, hashlib, random, re, os, time, datetime
from reedsolo import RSCodec
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# PHY & plotting libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob

# ---------- Base Station Config ----------
BS1_HOST, BS1_PORT = "127.0.0.1", 50050
BS2_HOST, BS2_PORT = "127.0.0.1", 50051

# ---------- Primary / Secondary setup ----------
PRIMARY_SENDERS = ["JAZZ", "UFONE", "TELENOR", "WARID", "STARLINK", "ZONG", "SCO", "PTCL"]

# ---------- Crypto / FEC ----------
PASSPHRASE = b"very secret passphrase - change this!"
KEY = hashlib.sha256(PASSPHRASE).digest()
rs = RSCodec(40)
stop_event = threading.Event()
plot_lock = threading.Lock()

# --- Session Management ---
chat_partner = None
pending_request_from = None
state_lock = threading.Lock()

# Control messages
CTL_CONNECT_REQUEST = "__CONNECT_REQUEST__"
CTL_CONNECT_ACCEPT = "__CONNECT_ACCEPT__"
CTL_CONNECT_REJECT = "__CONNECT_REJECT__"
CTL_DISCONNECT = "__DISCONNECT__"

# ---------- Sensing state ----------
# Per-peer sensing statistics for dynamic thresholding
sensing_stats = {}  # peer_id -> {"ema_energy": float, "alpha": 0.1, "min_floor": float}

# ---------- FIXED Helpers ----------
def aes_gcm_encrypt(plaintext, key):
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce + tag + ciphertext

def aes_gcm_decrypt(enc_blob, key):
    nonce, tag, ciphertext = enc_blob[:12], enc_blob[12:28], enc_blob[28:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

def determine_M(msg):
    return determine_M_local(msg)

def recv_full(sock, length, timeout=10.0):
    """FIXED: More robust receive with better timeout handling"""
    data = b""
    start_time = time.time()
    
    while len(data) < length and not stop_event.is_set():
        try:
            remaining = length - len(data)
            # Use smaller chunks for better responsiveness
            chunk_size = min(1024, remaining)
            sock.settimeout(0.5)  # Short timeout for responsiveness
            chunk = sock.recv(chunk_size)
            
            if not chunk:
                # Socket closed by peer
                raise ConnectionError("Socket closed by peer")
                
            data += chunk
            
            # Check overall timeout
            if time.time() - start_time > timeout:
                raise ConnectionError(f"Receive timeout after {timeout}s")
                
        except socket.timeout:
            # Expected for non-blocking reads, continue
            if stop_event.is_set():
                break
            continue
        except BlockingIOError:
            if stop_event.is_set():
                break
            time.sleep(0.01)
            continue
        except ConnectionError:
            raise
        except Exception as e:
            print(f"[RECV_ERROR] Unexpected error: {e}")
            raise ConnectionError(f"Receive error: {e}")
    
    return data if len(data) == length else None

# ---------- PHY helpers ----------
def bits_from_bytes(b: bytes):
    if b is None or len(b) == 0:
        return np.array([], dtype=np.uint8)
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

def qam_mod(bits, M):
    if M <= 1:
        return np.array([], dtype=np.complex128)
    k = int(np.log2(M))
    if bits.size == 0:
        return np.array([], dtype=np.complex128)
    # pad to multiple of k
    pad = (-bits.size) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    ints = bits.reshape((-1, k)).dot(1 << np.arange(k - 1, -1, -1))
    sqrtM = int(np.sqrt(M))
    x_int = (ints % sqrtM).astype(np.int64)
    y_int = (ints // sqrtM).astype(np.int64)
    raw_x = 2 * x_int - (sqrtM - 1)
    raw_y = 2 * y_int - (sqrtM - 1)
    scale = np.sqrt((2.0 / 3.0) * (M - 1)) if M > 1 else 1.0
    return (raw_x / scale) + 1j * (raw_y / scale)

def ofdm_mod(symbols, num_subcarriers, cp_len):
    if len(symbols) == 0:
        return np.array([], dtype=np.complex128)
    n_ofdm = int(np.ceil(len(symbols) / num_subcarriers))
    padded = np.pad(symbols, (0, n_ofdm * num_subcarriers - len(symbols)))
    reshaped = padded.reshape((n_ofdm, num_subcarriers))
    ifft_data = np.fft.ifft(reshaped, axis=1)
    ofdm_with_cp = np.hstack([ifft_data[:, -cp_len:], ifft_data])
    return ofdm_with_cp.flatten()

def make_constellation_plot(symbols, title, message_text, M, nc, cp):
    fig = plt.figure(figsize=(8,6))
    if symbols.size > 0:
        plt.scatter(np.real(symbols), np.imag(symbols), s=6, alpha=0.7)
    plt.title(title)
    plt.grid(True)
    plt.xlabel("I")
    plt.ylabel("Q")
    
    textstr = f'Message: "{message_text[:50]}{"..." if len(message_text)>50 else ""}"\nM: {M}\nSubcarriers: {nc}\nCP Length: {cp}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.figtext(0.02, 0.02, textstr, fontsize=9, bbox=props)
    
    return fig

def make_ofdm_plot(signal, title, message_text, M, nc, cp):
    fig = plt.figure(figsize=(8,4))
    if len(signal) > 0:
        plt.plot(np.real(signal[:300]), label="I", linewidth=1, alpha=0.8)
        plt.plot(np.imag(signal[:300]), label="Q", linewidth=1, alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    textstr = f'Message: "{message_text[:50]}{"..." if len(message_text)>50 else ""}"\nM: {M}\nSubcarriers: {nc}\nCP Length: {cp}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.figtext(0.02, 0.02, textstr, fontsize=9, bbox=props)
    
    return fig

def make_sensing_plot(sensing_data, title, message_text):
    """Enhanced sensing visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Energy comparison
    if sensing_data and 'test_statistic' in sensing_data:
        test_stat = sensing_data['test_statistic']
        threshold = sensing_data.get('threshold', test_stat * 0.6)
        busy = sensing_data.get('busy', False)
        
        bars = ax1.bar(['Energy', 'Threshold'], [test_stat, threshold], 
                      color=['red' if busy else 'green', 'orange'])
        ax1.set_ylabel('Power')
        ax1.set_title(f"Channel: {'BUSY' if busy else 'IDLE'}")
        ax1.grid(True, alpha=0.3)
    
    # Message info
    ax2.axis('off')
    info_text = f"""Message Info:
From: {title.split(' - ')[0] if ' - ' in title else 'Unknown'}
Length: {len(message_text)} chars
Modulation: QAM-{sensing_data.get('M', 'N/A')}
Subcarriers: {sensing_data.get('nc', 'N/A')}
Sensing Energy: {sensing_data.get('test_statistic', 0):.2e}
Threshold: {sensing_data.get('threshold', 0):.2e}
Busy: {sensing_data.get('busy', False)}"""
    
    ax2.text(0.03, 0.95, info_text, transform=ax2.transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

def save_plots_to_pdf(pdf_path, figs):
    with plot_lock:
        with PdfPages(pdf_path) as pdf:
            for f in figs:
                pdf.savefig(f)
                plt.close(f)

# ---------- Data storage ----------
DATA_DIR = "node_logs"
os.makedirs(DATA_DIR, exist_ok=True)

messages_by_peer = {}
base_station_events = []

def cleanup_old_files(node_id):
    pattern = os.path.join(DATA_DIR, f"{node_id}_*")
    old_files = glob.glob(pattern)
    for file_path in old_files:
        try:
            os.remove(file_path)
            print(f"[CLEANUP] Removed old file: {file_path}")
        except Exception as e:
            print(f"[CLEANUP] Error removing {file_path}: {e}")

def log_event(txt):
    t = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
    base_station_events.append(f"{t} {txt}")

def update_sensing_stats(peer_id, energy):
    """
    Update per-peer sensing statistics (exponential moving average).
    Returns (threshold, busy_bool, ema_energy).
    """
    if peer_id not in sensing_stats:
        sensing_stats[peer_id] = {"ema_energy": max(energy, 1e-12), "alpha": 0.12, "min_floor": 1e-12}
    stats = sensing_stats[peer_id]
    alpha = stats.get("alpha", 0.12)
    # Update EMA for noise floor estimate
    ema_prev = stats.get("ema_energy", max(energy, 1e-12))
    ema_new = (1 - alpha) * ema_prev + alpha * max(energy, 1e-12)
    stats["ema_energy"] = ema_new
    # threshold factor: slightly above noise floor
    threshold = max(stats["min_floor"], ema_new * 2.5)
    busy = energy > threshold
    return threshold, busy, ema_new

def record_message(peer_id, text_bytes, M, message_text, sensing_data=None):
    entry = {"ts": time.time(), "bytes": text_bytes, "M": M, "text": message_text, "sensing_data": sensing_data}
    try:
        bits = bits_from_bytes(text_bytes)
        tx_symbols = qam_mod(bits, M)
        nc, cp = pick_num_subcarriers_local(text_bytes, M)
        ofdm_sig = ofdm_mod(tx_symbols, nc, cp)
        entry["nc"] = nc
        entry["cp"] = cp
        entry["tx_symbols"] = tx_symbols
        entry["ofdm_sig"] = ofdm_sig
        
        # Compute actual energy from the real OFDM signal
        energy = np.mean(np.abs(ofdm_sig)**2) if ofdm_sig.size > 0 else 0.0
        
        # Update sensing stats for this peer using actual energy
        threshold, busy, ema = update_sensing_stats(peer_id, energy)
        entry["sensing_data"] = {
            "test_statistic": energy,
            "threshold": threshold,
            "busy": busy,
            "M": M,
            "nc": nc,
            "cp": cp,
            "snr_estimate": estimate_snr_from_energy(energy, M)  # simple SNR heuristic
        }
    except Exception as e:
        print(f"[PHY_ERROR] Failed to process PHY data: {e}")
        entry.update({"nc": 0, "cp": 0, "tx_symbols": np.array([]), "ofdm_sig": np.array([])})
    
    messages_by_peer.setdefault(peer_id, []).append(entry)
    print(f"[PHY_PARAMS] Received from {peer_id}: M={M}, NC={entry['nc']}, CP={entry['cp']}, Energy={entry.get('sensing_data',{}).get('test_statistic','N/A')}")

def estimate_snr_from_energy(energy, M):
    """
    Very simple heuristic to estimate SNR (dB) from energy.
    This is not a rigorous estimator but gives a working value for plotting/analysis.
    """
    # avoid zero division
    if energy <= 0:
        return -50.0
    # scale energy into a pseudo SNR
    snr_linear = max(1e-9, energy * 10.0)
    snr_db = 10.0 * np.log10(snr_linear)
    return float(np.clip(snr_db, -50.0, 60.0))

# ---------- Local decision helpers ----------
def determine_M_local(msg_or_path: str) -> int:
    if os.path.isfile(msg_or_path):
        ext = os.path.splitext(msg_or_path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]: return 256
        if ext in [".wav", ".mp3"]: return 64
    n = len(msg_or_path)
    if n <= 100: return 16
    if n <= 500: return 64
    return 256

def pick_num_subcarriers_local(encoded_bytes_or_text, M):
    candidates = [16, 32, 64, 128]
    lat, mob = [], []
    try:
        if isinstance(encoded_bytes_or_text, (bytes, bytearray)):
            bits = bits_from_bytes(encoded_bytes_or_text)
        else:
            bits = bits_from_bytes(str(encoded_bytes_or_text).encode('utf-8'))
        tx_symbols = qam_mod(bits, M)
    except Exception:
        tx_symbols = qam_mod(np.zeros(2, dtype=np.uint8), 16)
    
    for nc in candidates:
        sig = ofdm_mod(tx_symbols, nc, max(4, nc//8))
        lat.append(len(sig))
        mob.append(1.0/nc)
    
    lat, mob = np.array(lat), np.array(mob)
    lat_n = (lat - lat.min()) / (np.ptp(lat) + 1e-12)
    mob_n = (mob - mob.min()) / (np.ptp(mob) + 1e-12)
    cost = 0.7 * lat_n + 0.3 * mob_n
    idx = int(np.argmin(cost))
    nc, cp = candidates[idx], max(4, candidates[idx]//8)
    return nc, cp

# ---------- FIXED Message Handlers ----------
def compute_ofdm_energy_from_message_bytes(message_bytes, M, nc, cp):
    bits = bits_from_bytes(message_bytes)
    tx_syms = qam_mod(bits, M)
    ofdm_sig = ofdm_mod(tx_syms, nc, cp)
    energy = np.mean(np.abs(ofdm_sig)**2) if ofdm_sig.size > 0 else 0.0
    return energy, ofdm_sig

def send_message(sock, recipient, message_text):
    global chat_partner
    try:
        M = determine_M(message_text)
        nc, cp = pick_num_subcarriers_local(message_text, M)
        msg_bytes = message_text.encode("utf-8")
        energy, ofdm_sig = compute_ofdm_energy_from_message_bytes(msg_bytes, M, nc, cp)

        control_msgs = [CTL_CONNECT_REQUEST, CTL_CONNECT_ACCEPT, CTL_CONNECT_REJECT, CTL_DISCONNECT]
        
        pkt = struct.pack(">H", len(message_text.encode())) + message_text.encode()
        plaintext = struct.pack(">H H B", M, nc, cp) + pkt
        enc = aes_gcm_encrypt(plaintext, KEY)
        enc_rs = rs.encode(enc)
        
        sensing_header = struct.pack(">d", float(energy))
        dst_b = recipient.encode("utf-8")
        wire_payload = struct.pack(">H", len(dst_b)) + dst_b + sensing_header + enc_rs
        
        sock.sendall(struct.pack(">I", len(wire_payload)) + wire_payload)

        # Record message based on real bits/symbols
        if message_text not in control_msgs:
            # update our local records using the actual bytes and energy
            threshold, busy, ema = update_sensing_stats(recipient, energy)
            record_message(recipient, msg_bytes, M, message_text, {
                "test_statistic": energy,
                "threshold": threshold,
                "busy": busy,
                "M": M,
                "nc": nc,
                "cp": cp,
                "snr_estimate": estimate_snr_from_energy(energy, M)
            })
        
        log_event(f"Sent to {recipient}: {message_text}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to send message: {e}")
        log_event(f"Send error: {e}")
        with state_lock:
            if chat_partner == recipient:
                chat_partner = None
        return False

def decode_message_packet(data):
    """FIXED: Robust packet decoding that handles base station forwarding"""
    try:
        # The base station forwards packets with: [src_len][src_id][sensing_header?][encrypted_payload]
        if len(data) < 2:
            return False, None, None, None, None, None, None
            
        src_len = struct.unpack(">H", data[:2])[0]
        if len(data) < 2 + src_len:
            return False, None, None, None, None, None, None
            
        src_id = data[2:2+src_len].decode("utf-8")
        pos = 2 + src_len
        
        # Check for sensing header (8 bytes)
        sensing_energy = None
        if len(data[pos:]) >= 8:
            try:
                sensing_energy = struct.unpack(">d", data[pos:pos+8])[0]
                pos += 8
            except:
                # If sensing header decode fails, continue without it
                sensing_energy = 0.0
                pass
        
        # Remaining data is the encrypted payload
        enc_blob = data[pos:]
        
        if len(enc_blob) == 0:
            return False, None, None, None, None, None, None
        
        # Try RS decode
        try:
            rs_decoded = rs.decode(enc_blob)[0]
        except Exception as e:
            print(f"[DECODE_ERROR] RS decode failed: {e}")
            return False, None, None, None, None, None, None
        
        # Try AES decrypt
        try:
            plaintext = aes_gcm_decrypt(rs_decoded, KEY)
        except Exception as e:
            print(f"[DECODE_ERROR] AES decrypt failed: {e}")
            return False, None, None, None, None, None, None
        
        # Parse plaintext: [M][nc][cp][msg_len][message]
        if len(plaintext) < 5:
            return False, None, None, None, None, None, None
            
        try:
            M, nc, cp = struct.unpack(">H H B", plaintext[:5])
            pkt = plaintext[5:]
            
            if len(pkt) < 2:
                return False, None, None, None, None, None, None
                
            msg_len = struct.unpack(">H", pkt[:2])[0]
            
            if len(pkt) < 2 + msg_len:
                return False, None, None, None, None, None, None
                
            msg = pkt[2:2+msg_len].decode("utf-8")
            
            return True, src_id, msg, M, nc, cp, sensing_energy
            
        except Exception as e:
            print(f"[DECODE_ERROR] Plaintext parsing failed: {e}")
            return False, None, None, None, None, None, None
            
    except Exception as e:
        print(f"[DECODE_ERROR] General decode error: {e}")
        return False, None, None, None, None, None, None

def receive_handler(sock):
    """FIXED: Stable receive handler with connection management"""
    global chat_partner, pending_request_from
    
    print("[RECEIVE] Receive handler started")
    
    while not stop_event.is_set():
        try:
            # Set a reasonable socket timeout
            sock.settimeout(1.0)
            
            # Read message length header (4 bytes)
            hdr_data = recv_full(sock, 4, timeout=30.0)
            if not hdr_data:
                if stop_event.is_set():
                    break
                # No data available, continue polling
                continue
                
            length = struct.unpack(">I", hdr_data)[0]
            
            # Sanity check on message length
            if length == 0 or length > 100000:
                print(f"[PROTOCOL] Invalid message length: {length}")
                continue
            
            # Read the message payload
            message_data = recv_full(sock, length, timeout=30.0)
            if not message_data:
                print("[PROTOCOL] Failed to receive message body")
                continue
            
            # Decode the message
            success, src_id, msg, M, nc, cp, sensing_energy = decode_message_packet(message_data)
            
            if not success:
                print(f"[DECODE] Failed to decode message ({len(message_data)} bytes)")
                continue
            
            print(f"[RECEIVE] Successfully decoded message from {src_id}: '{msg}'")
            
            # If the sender included a sensing energy header, prefer that; otherwise compute from message
            if sensing_energy is None:
                # compute energy using actual bytes from payload message
                try:
                    # message text length is known - reconstruct raw plaintext bytes for energy computation:
                    # but we don't have original plaintext bytes here; we can compute energy from the plaintext msg bytes
                    msg_bytes = msg.encode("utf-8")
                    energy, ofdm_sig = compute_ofdm_energy_from_message_bytes(msg_bytes, M, nc, cp)
                except Exception:
                    energy = 0.0
            else:
                energy = sensing_energy
            
            # Update sensing stats using the actual energy
            threshold, busy, ema = update_sensing_stats(src_id, energy)
            
            # Create sensing data structure
            sensing_data = {
                'test_statistic': energy,
                'M': M, 'nc': nc, 'cp': cp,
                'busy': busy,
                'threshold': threshold,
                'snr_estimate': estimate_snr_from_energy(energy, M)
            }
            
            # Handle control messages
            if msg == CTL_CONNECT_REQUEST:
                with state_lock:
                    if not chat_partner and not pending_request_from:
                        pending_request_from = src_id
                        print(f"\n[!] Connection request from {src_id}. Type 'accept' or 'reject'.\n> ", end="", flush=True)
                        log_event(f"Connection request from {src_id}")
                continue
            
            elif msg == CTL_CONNECT_ACCEPT:
                with state_lock:
                    if pending_request_from == src_id:
                        pending_request_from = None
                        chat_partner = src_id
                        print(f"\n[SUCCESS] {src_id} accepted your connection request!")
                        print(f"You can now chat with {src_id}.")
                        print(f"> ", end="", flush=True)
                continue
                
            elif msg == CTL_CONNECT_REJECT:
                with state_lock:
                    if pending_request_from == src_id:
                        pending_request_from = None
                        print(f"\n[INFO] {src_id} rejected your connection request.")
                        print(f"> ", end="", flush=True)
                continue
            
            elif msg == CTL_DISCONNECT:
                with state_lock:
                    if src_id == chat_partner:
                        print(f"\n[INFO] {chat_partner} has disconnected.")
                        chat_partner = None
                        log_event(f"Disconnected by {src_id}")
                continue

            # Handle regular messages
            with state_lock:
                if src_id == chat_partner:
                    print(f"\n<<< {src_id}: '{msg}'\n> ", end="", flush=True)
                    record_message(src_id, msg.encode("utf-8"), M, msg, sensing_data)
                    log_event(f"Received message from {src_id}")
                else:
                    print(f"\n[INFO] Message from {src_id} (not partner): '{msg}'")
                    record_message(src_id, msg.encode("utf-8"), M, msg, sensing_data)

        except socket.timeout:
            # Expected - no data available
            continue
        except ConnectionError as e:
            if not stop_event.is_set():
                print(f"\n[CONNECTION] Connection lost: {e}")
                break
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n[RECEIVE_ERROR] Unexpected error: {e}")
                # Continue running despite errors
                time.sleep(1)
    
    print("[RECEIVE] Receive handler stopped")

def main_loop(sock):
    global chat_partner, pending_request_from
    
    print(f"\nReceiver started. Type 'exit' to quit.")
    
    while not stop_event.is_set():
        try:
            with state_lock:
                current_partner = chat_partner
                current_request = pending_request_from
            
            # Determine prompt
            if current_partner:
                prompt = f"Chat with {current_partner} > "
            elif current_request:
                prompt = f"Connection request from {current_request} (accept/reject): "
            else:
                prompt = "Waiting for connections... (type 'exit' to quit): "
            
            # Get user input
            try:
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] Shutting down...")
                stop_event.set()
                break

            if not user_input:
                continue

            if user_input.lower() == 'exit':
                if current_partner:
                    send_message(sock, current_partner, CTL_DISCONNECT)
                stop_event.set()
                break

            # Handle connection requests
            if current_request:
                if user_input.lower() == 'accept':
                    if send_message(sock, current_request, CTL_CONNECT_ACCEPT):
                        with state_lock:
                            chat_partner = current_request
                            pending_request_from = None
                        print(f"[SUCCESS] Connected with {chat_partner}. You can now chat.")
                        log_event(f"Accepted connection from {current_request}")
                    else:
                        print("[ERROR] Failed to send acceptance")
                else:  # reject
                    send_message(sock, current_request, CTL_CONNECT_REJECT)
                    with state_lock:
                        pending_request_from = None
                    print("[INFO] Connection request rejected.")
            
            # Handle chat messages
            elif current_partner:
                if not send_message(sock, current_partner, user_input):
                    print("[ERROR] Failed to send message")
                    with state_lock:
                        chat_partner = None
            
            # Idle state
            else:
                print("[INFO] Not in a chat session. Waiting for connection requests...")

        except Exception as e:
            print(f"[LOOP_ERROR] Error in main loop: {e}")
            if not stop_event.is_set():
                time.sleep(0.5)

# ---------- Plot Generation & Export ----------
def compute_ber_for_snr(tx_bits, M, snr_db):
    """
    Compute BER given actual transmitted bits (tx_bits) using an AWGN model.
    Note: this uses actual tx_bits (not random), then corrupts with noise for BER simulation.
    """
    if tx_bits is None or tx_bits.size == 0:
        return 1.0
        
    symbols = qam_mod(tx_bits, M)
    if symbols.size == 0:
        return 1.0
    snr_linear = 10**(snr_db/10.0)
    noise_var = 1.0 / (2 * snr_linear)
    # noise generation required for evaluation; this is simulation noise (not sensing)
    noise = (np.sqrt(noise_var) * (np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape)))
    rx = symbols + noise
    
    # symbol error approximation
    symbol_errors = np.mean(np.abs(rx - symbols) > 0.5)
    k = int(np.log2(max(2, M)))
    ber = min(1.0, symbol_errors * k / 2.0)
    
    return ber

def export_all_results(node_id):
    figs = []
    
    # Message visualizations
    for peer, entries in messages_by_peer.items():
        for i, entry in enumerate(entries):
            tx_syms = entry.get("tx_symbols", np.array([]))
            ofdm_sig = entry.get("ofdm_sig", np.array([]))
            message_text = entry.get("text", "Unknown message")
            M, nc, cp = entry.get("M", 16), entry.get("nc", 64), entry.get("cp", 8)
            sensing_data = entry.get("sensing_data", {})
            
            if tx_syms.size > 0:
                figs.append(make_constellation_plot(tx_syms, f"{peer} - Constellation ({i})", 
                                                  message_text, M, nc, cp))
                figs.append(make_ofdm_plot(ofdm_sig, f"{peer} - OFDM Signal ({i})", 
                                         message_text, M, nc, cp))
            
            if sensing_data:
                figs.append(make_sensing_plot(sensing_data, f"{peer} - Spectrum Sensing ({i})", message_text))
    
    # Save plots
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pdf_path = os.path.join(DATA_DIR, f"{node_id}_plots_{timestamp}.pdf")
    
    try:
        save_plots_to_pdf(pdf_path, figs)
        print(f"[EXPORT] Saved {len(figs)} plots to {pdf_path}")
    except Exception as e:
        print(f"[EXPORT_ERROR] Failed to save PDF: {e}")
    
    # Save log file
    log_path = os.path.join(DATA_DIR, f"{node_id}_log_{timestamp}.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"5G Communication Pipeline - Receiver Node {node_id}\n")
            f.write(f"Report Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')}\n")
            f.write("="*60 + "\n\n")
            
            f.write("EVENT LOG:\n")
            f.write("-" * 40 + "\n")
            for event in base_station_events:
                f.write(event + "\n")
            
            f.write("\n\nMESSAGE LOG:\n")
            f.write("=" * 60 + "\n")
            for peer, entries in messages_by_peer.items():
                f.write(f"\nPEER: {peer}\n")
                f.write("-" * 40 + "\n")
                for i, entry in enumerate(entries):
                    f.write(f"Message {i+1}:\n")
                    f.write(f"  Time: {datetime.datetime.fromtimestamp(entry['ts'], datetime.timezone.utc).isoformat().replace('+00:00', 'Z')}\n")
                    f.write(f"  Modulation: QAM-{entry.get('M', 'N/A')}\n")
                    f.write(f"  Subcarriers: {entry.get('nc', 'N/A')}, CP: {entry.get('cp', 'N/A')}\n")
                    f.write(f"  Energy: {entry.get('sensing_data', {}).get('test_statistic', 'N/A')}\n")
                    f.write(f"  Threshold: {entry.get('sensing_data', {}).get('threshold', 'N/A')}\n")
                    f.write(f"  Busy: {entry.get('sensing_data', {}).get('busy', 'N/A')}\n")
                    f.write(f"  Content: {entry.get('text', 'Unknown')}\n\n")
        
        print(f"[EXPORT] Saved log to {log_path}")
        log_event(f"Exported results to {pdf_path} and {log_path}")
        
    except Exception as e:
        print(f"[EXPORT_ERROR] Failed to save log: {e}")

# ---------- Main ----------
def main():
    # Get receiver ID
    node_id = ""
    while not re.match(r"^R([1-9]|[1-5][0-9]|60)$", node_id):
        node_id = input("Enter your Receiver ID (R1-R60): ").strip().upper()
    
    cleanup_old_files(node_id)
    
    # Determine base station
    node_num = int(node_id[1:])
    if 1 <= node_num <= 30:
        BS_HOST, BS_PORT = BS1_HOST, BS1_PORT
        node_pos = (random.uniform(10, 140), random.uniform(-50, 50))
    else:
        BS_HOST, BS_PORT = BS2_HOST, BS2_PORT  
        node_pos = (random.uniform(210, 340), random.uniform(-50, 50))
    
    # Connect to base station
    try:
        sock = socket.create_connection((BS_HOST, BS_PORT), timeout=10)
        sock.settimeout(2.0)  # Reasonable socket timeout
        
        reg_msg = {
            "type": "receiver", 
            "id": node_id, 
            "pos": list(node_pos)
        }
        sock.sendall((json.dumps(reg_msg) + "\n").encode("utf-8"))
        
        print(f"RECEIVER '{node_id}' successfully registered with Base Station")
        print(f"Position: {node_pos}")
        log_event(f"Registered {node_id} at {node_pos}")
        
    except Exception as e:
        print(f"Failed to connect to Base Station: {e}")
        return
    
    # Start receive handler
    receive_thread = threading.Thread(target=receive_handler, args=(sock,), daemon=True)
    receive_thread.start()
    
    # Main interaction loop
    try:
        main_loop(sock)
    except Exception as e:
        print(f"\n[MAIN_ERROR] Fatal error: {e}")
    finally:
        # Clean shutdown
        stop_event.set()
        try:
            export_all_results(node_id)
            sock.close()
        except:
            pass
        print(f"\nRECEIVER '{node_id}': Shutdown complete.")

if __name__ == "__main__":
    main()
