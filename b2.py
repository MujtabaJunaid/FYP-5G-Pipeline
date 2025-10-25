# base_station.py (updated)
# - Base station with primary/secondary priority
# - Energy-based spectrum sensing using actual message bits (BS decrypts message to construct bits)
# - If primary recent activity detected, secondary transmissions will be queued (opportunistic)
# - Generates PDF report including sensing energy timeline and busy/idle shading

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

# --- CONFIGURATION ---
# Change this value to 1 or 2 to run as BS1 or BS2
BS_INSTANCE = 2
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

# ---------- Crypto & RS (so BS can inspect message bits for sensing) ----------
PASSPHRASE = b"very secret passphrase - change this!"
KEY = hashlib.sha256(PASSPHRASE).digest()
rs = RSCodec(40)

# ---------- Primary list ----------
PRIMARY_SENDERS = ["JAZZ", "UFONE", "TELENOR", "WARID", "STARLINK", "ZONG", "SCO", "PTCL"]

# ---------- Sensing state ----------
ENERGY_HISTORY = []  # list of (ts, src_id, energy)
LAST_PRIMARY_ACTIVITY = 0.0
PRIMARY_PROTECTION_WINDOW = 10.0  # seconds; after a primary transmission, protect channel for this many seconds

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
    except Exception:
        return False

# ---------- PHY helpers to rebuild OFDM energy from message bits ----------
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

def process_incoming_frame(src_id, dst_id, encoded_bytes, hop_count=0):
    """Process a frame incoming from a local client. This function now:
       - tries to RS-decode & AES-decrypt the payload to access plaintext (for sensing)
       - computes message OFDM energy (from message bits)
       - updates energy history and primary activity timestamp
       - if sender is secondary and primary activity recent -> queue the message
       - otherwise attempt local delivery or hop to neighbor
    """
    # Attempt to parse a sensing header if present (we expect senders put an 8-byte energy after dst id, before RS)
    # encoded_bytes starts at the first byte after destination id in the original payload.
    # We'll try to extract the first 8 bytes as a double if it's present and plausible. But encoded_bytes could be RS encoded.
    # Safer approach: try to RS.decode then AES decrypt to get plaintext and compute energy based on message bits.
    try:
        rs_decoded = rs.decode(encoded_bytes)[0]
        plaintext = aes_gcm_decrypt(rs_decoded, KEY)
    except Exception:
        # Can't decode/decrypt — fallback: treat as unknown energy 0 and forward as before
        plaintext = None

    energy = 0.0
    ofdm_sig = np.array([])
    M = 16; nc = 64; cp = 8; msg_text = "<unknown>"

    if plaintext is not None:
        energy, ofdm_sig, M, nc, cp, msg_text = compute_energy_from_plaintext(plaintext)
        # update global energy history
        ENERGY_HISTORY.append((time.time(), src_id, energy))
        # if src is primary, update LAST_PRIMARY_ACTIVITY
        if src_id.upper() in PRIMARY_SENDERS:
            global LAST_PRIMARY_ACTIVITY
            LAST_PRIMARY_ACTIVITY = time.time()
            log_msg = f"Primary TX detected from {src_id}, energy={energy:.4e}"
            print(f"[BS {BS_ID}] {log_msg}")
            with stats_lock:
                stats.setdefault("primary_events", 0)
                stats["primary_events"] += 1
        else:
            # log secondary energy
            print(f"[BS {BS_ID}] Secondary TX from {src_id}, energy={energy:.4e}")

    # Enforce priority: if this sender is secondary and a primary was active recently, queue
    is_primary_sender = (src_id.upper() in PRIMARY_SENDERS)
    now = time.time()
    if (not is_primary_sender) and (now - LAST_PRIMARY_ACTIVITY) < PRIMARY_PROTECTION_WINDOW:
        # queue the message for retry rather than forwarding now
        print(f"[BS {BS_ID}] Queuing secondary message from {src_id} (recent primary activity).")
        with stats_lock:
            stats["queued"] += 1
        item = {"src_id": src_id, "dst_id": dst_id, "payload": encoded_bytes, "hop_count": hop_count, "ts": time.time()}
        with retry_queue_lock:
            retry_queue.append(item)
        return

    # If we reach here, we attempt immediate delivery
    with clients_lock:
        dst_info = clients.get(dst_id)
    if dst_info and distance(BS_POS, dst_info.get("pos", (0,0))) <= COMM_RANGE:
        # Prepend the source ID so the client knows who sent the message.
        src_bytes = src_id.encode("utf-8")
        payload = struct.pack(">H", len(src_bytes)) + src_bytes + encoded_bytes
        wire_frame = struct.pack(">I", len(payload)) + payload

        if send_all_with_retry(dst_info["conn"], wire_frame):
            print(f"[BS {BS_ID}] Delivered from {src_id} -> local {dst_id}")
            with stats_lock:
                stats["forwarded_local"] += 1; stats["delivered"] += 1
                stats["per_hop_counts"].append(hop_count)
            return

    # Not local or couldn't deliver: hop to neighbor(s)
    src_bytes, dst_bytes = src_id.encode("utf-8"), dst_id.encode("utf-8")
    hop_payload = (struct.pack(">B", hop_count + 1) +
                   struct.pack(">H", len(src_bytes)) + src_bytes +
                   struct.pack(">H", len(dst_bytes)) + dst_bytes + encoded_bytes)
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
        except Exception: continue

    if not forwarded:
        print(f"[BS {BS_ID}] Queued message for {dst_id} (no neighbor available)")
        with stats_lock: stats["queued"] += 1
        item = {"src_id": src_id, "dst_id": dst_id, "payload": encoded_bytes, "hop_count": hop_count, "ts": time.time()}
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
            remaining = payload[2+dst_len:]

            # If sender included a sensing header (8 bytes) before RS-coded payload, attempt to extract
            sensing_energy = None
            if len(remaining) >= 8:
                # try unpack double and assume that's sensing header; if it doesn't decode later, it's okay
                try:
                    sensing_energy = struct.unpack(">d", remaining[:8])[0]
                    encoded_bytes = remaining[8:]
                except Exception:
                    sensing_energy = None
                    encoded_bytes = remaining
            else:
                encoded_bytes = remaining

            with stats_lock: stats["received"] += 1
            threading.Thread(target=process_incoming_frame, args=(client_id, dst_id, encoded_bytes), daemon=True).start()

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

# ---------- Retry thread to attempt queued deliveries periodically ----------
def retry_worker():
    while not STOP.is_set():
        try:
            with retry_queue_lock:
                if not retry_queue:
                    pass
                else:
                    item = retry_queue.popleft()
                    # check if primary protection window expired; if not, push back
                    if (time.time() - LAST_PRIMARY_ACTIVITY) < PRIMARY_PROTECTION_WINDOW:
                        retry_queue.append(item)
                        time.sleep(0.5)
                        continue
                    # try deliver
                    process_incoming_frame(item["src_id"], item["dst_id"], item["payload"], item.get("hop_count", 0))
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
            continue

# ---------- Reporting (energy timeline plot) ----------
def export_report():
    try:
        figs = []
        # Energy timeline plot
        if ENERGY_HISTORY:
            ts = np.array([e[0] for e in ENERGY_HISTORY])
            energies = np.array([e[2] for e in ENERGY_HISTORY])
            labels = [e[1] for e in ENERGY_HISTORY]
            fig = plt.figure(figsize=(10,4))
            plt.plot(ts - ts[0], energies, marker='o', linewidth=1)
            # Shade regions where primary activity recorded within protection window
            for i,(t, src, en) in enumerate(ENERGY_HISTORY):
                if src.upper() in PRIMARY_SENDERS:
                    # shade a rectangle for protection window
                    start = t - ts[0] - 0.1
                    end = start + PRIMARY_PROTECTION_WINDOW
                    plt.axvspan(max(0,start), end, alpha=0.12, color='red')
            plt.title(f"{BS_ID} - Energy timeline (per-message). Red shading = primary protection windows")
            plt.xlabel("Time (s since first measured)")
            plt.ylabel("Avg OFDM energy (power)")
            figs.append(fig)
        else:
            fig = plt.figure(figsize=(8,4)); plt.text(0.5, 0.5, "No energy measurements", ha='center', va='center', transform=plt.gca().transAxes); plt.title("Energy timeline"); figs.append(fig)

        # Stats summary figure
        fig2 = plt.figure(figsize=(6,4))
        txt = f"Stats - {BS_ID}\nReceived: {stats.get('received',0)}\nForwarded Local: {stats.get('forwarded_local',0)}\nForwarded Remote: {stats.get('forwarded_remote',0)}\nQueued: {stats.get('queued',0)}\nDelivered: {stats.get('delivered',0)}\nPrimary Events: {stats.get('primary_events',0)}"
        plt.text(0.01, 0.5, txt, fontsize=12, va='center')
        plt.axis('off')
        figs.append(fig2)

        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        pdf_path = os.path.join(RESULTS_DIR, f"base_station_{BS_INSTANCE}_report_{timestamp}.pdf")
        with PdfPages(pdf_path) as pdf:
            for f in figs:
                pdf.savefig(f)
                plt.close(f)
        print(f"[BS {BS_ID}] Exported report to {pdf_path}")
    except Exception as e:
        print(f"[BS {BS_ID}] Error exporting report: {e}")

def main():
    # start retry worker
    threading.Thread(target=retry_worker, daemon=True).start()
    try:
        accept_loop()
    except KeyboardInterrupt:
        print(f"\n[BS {BS_ID}] Shutting down...")
    finally:
        STOP.set()
        # export report before exit
        export_report()

if __name__ == "__main__":
    main()
