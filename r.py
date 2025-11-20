# Dynamic_Receiver.py  (updated)
# Enhanced dynamic receiver client with:
# - Connection handshake (accept/reject) and persistent chat sessions
# - disconnect detection
# - message & PHY data logging
# - constellation, OFDM, I/Q plots
# - SNR vs BER simulation and export to PDF/log
# - Spectrum sensing plot per received message (uses message bits not random bits)
# - Primary/Secondary awareness (BS will enforce priority, receiver logs it)

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

# ---------- Primary / Secondary setup (for logging/awareness) ----------
# Mapping based on ID ranges:
# S1-S10: JAZZ, S11-S20: WARID, S21-S30: UFONE, S31-S40: TELENOR, S41-S50: ZONG
# S51-S60: SECONDARY (Opportunistic)

def get_node_type(node_id):
    """Returns (is_primary, operator_name) based on ID range."""
    if not node_id or not node_id.startswith("S"):
        return False, "UNKNOWN"
    try:
        num = int(node_id[1:])
        if 1 <= num <= 10: return True, "JAZZ"
        if 11 <= num <= 20: return True, "WARID"
        if 21 <= num <= 30: return True, "UFONE"
        if 31 <= num <= 40: return True, "TELENOR"
        if 41 <= num <= 50: return True, "ZONG"
        if 51 <= num <= 60: return False, "SECONDARY"
    except ValueError:
        pass
    return False, "UNKNOWN"

NODE_ID = None # Will be set in main

# ---------- Crypto / FEC ----------
PASSPHRASE = b"very secret passphrase - change this!"
KEY = hashlib.sha256(PASSPHRASE).digest()
rs = RSCodec(40)
stop_event = threading.Event()
plot_lock = threading.Lock()

# --- Session Management ---
# State variables to manage who we are talking to
chat_partner = None # The peer we are currently TYPING to (for replies)
connected_peers = set() # Set of ALL peers we have accepted
pending_request_from = None
state_lock = threading.Lock()

# Control messages for session management
CTL_CONNECT_REQUEST = "__CONNECT_REQUEST__"
CTL_CONNECT_ACCEPT = "__CONNECT_ACCEPT__"
CTL_CONNECT_REJECT = "__CONNECT_REJECT__"
CTL_DISCONNECT = "__DISCONNECT__"

# ---------- Helpers ----------
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

def recv_full(sock, length):
    data = b""
    while len(data) < length:
        try:
            more = sock.recv(length - len(data))
            if not more:
                raise ConnectionError("socket closed")
            data += more
        except socket.timeout:
            return None
    return data

# ---------- PHY helpers (from your snippet) ----------
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

def make_constellation_plot(symbols, title, message_text, M, nc, cp):
    fig = plt.figure(figsize=(8,6))
    plt.scatter(np.real(symbols), np.imag(symbols), s=6)
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
        plt.plot(np.real(signal[:300]), label="I", linewidth=1)
        plt.plot(np.imag(signal[:300]), label="Q", linewidth=1)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    textstr = f'Message: "{message_text[:50]}{"..." if len(message_text)>50 else ""}"\nM: {M}\nSubcarriers: {nc}\nCP Length: {cp}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.figtext(0.02, 0.02, textstr, fontsize=9, bbox=props)
    
    return fig

def make_sensing_plot(ofdm_sig, title, message_text, threshold=None):
    """Plot per-sample instantaneous power and show threshold and busy/idle."""
    fig = plt.figure(figsize=(8,3))
    power = np.abs(ofdm_sig)**2 if ofdm_sig.size>0 else np.array([])
    if power.size>0:
        t = np.arange(len(power))
        plt.plot(t[:1000], power[:1000], linewidth=1)
        if threshold is not None:
            plt.hlines(threshold, t[0], t[min(999,len(t)-1)], linestyles='--')
            busy = np.mean(power) > threshold
            plt.title(f"{title}  |  {'BUSY' if busy else 'IDLE'} (avg power={np.mean(power):.4e})")
        else:
            plt.title(title)
    else:
        plt.title(title + " (no signal)")
    plt.xlabel("Sample index")
    plt.ylabel("Instantaneous power")
    plt.grid(True)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.figtext(0.02, 0.02, f'Message: "{message_text[:50]}{"..." if len(message_text)>50 else ""}"', fontsize=9, bbox=props)
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
    """Delete old log files for this node"""
    pattern = os.path.join(DATA_DIR, f"{node_id}_*")
    old_files = glob.glob(pattern)
    for file_path in old_files:
        try:
            os.remove(file_path)
            print(f"[CLEANUP] Removed old file: {file_path}")
        except Exception as e:
            print(f"[CLEANUP] Error removing {file_path}: {e}")

def log_event(txt):
    t = datetime.datetime.utcnow().isoformat() + "Z"
    base_station_events.append(f"{t} {txt}")

def record_message(peer_id, text_bytes, M, message_text):
    """Record message and derive PHY objects for plotting and BER simulation."""
    entry = {"ts": time.time(), "bytes": text_bytes, "M": M, "text": message_text}
    try:
        bits = bits_from_bytes(text_bytes)
        tx_symbols = qam_mod(bits, M)
        nc, cp = pick_num_subcarriers_local(text_bytes, M)
        ofdm_sig = ofdm_mod(tx_symbols, nc, cp)
        entry["nc"] = nc
        entry["cp"] = cp
    except Exception:
        tx_symbols, ofdm_sig = np.array([]), np.array([])
        entry["nc"], entry["cp"] = 0, 0
    entry["tx_symbols"] = tx_symbols
    entry["ofdm_sig"] = ofdm_sig
    messages_by_peer.setdefault(peer_id, []).append(entry)
    print(f"[PHY_PARAMS] Received message with M={M}, Subcarriers={entry['nc']}, CP={entry['cp']}")

# ---------- Agent helpers (local versions from your snippet) ----------
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
        bits = bits_from_bytes(encoded_bytes_or_text) if isinstance(encoded_bytes_or_text, (bytes, bytearray)) else bits_from_bytes(str(encoded_bytes_or_text).encode('utf-8'))
        tx_symbols = qam_mod(bits, M)
    except Exception:
        tx_symbols = qam_mod(np.zeros(2, dtype=np.uint8), 16)
    for nc in candidates:
        sig = ofdm_mod(tx_symbols, nc, max(4,nc//8))
        lat.append(len(sig)); mob.append(1.0/nc)
    lat, mob = np.array(lat), np.array(mob)
    lat_n = (lat - lat.min()) / (np.ptp(lat) + 1e-12)
    mob_n = (mob - mob.min()) / (np.ptp(mob) + 1e-12)
    cost = 0.7 * lat_n + 0.3 * mob_n
    idx = int(np.argmin(cost))
    nc, cp = candidates[idx], max(4, candidates[idx]//8)
    return nc, cp

# ---------- Receive / Reply Handlers ----------
def compute_ofdm_energy_from_message_bytes(message_bytes, M, nc, cp):
    """Build OFDM from message bytes and compute average energy (power)."""
    bits = bits_from_bytes(message_bytes)
    tx_syms = qam_mod(bits, M)
    ofdm_sig = ofdm_mod(tx_syms, nc, cp)
    energy = np.mean(np.abs(ofdm_sig)**2) if ofdm_sig.size>0 else 0.0
    return energy, ofdm_sig

def sense_channel_environment():
    """
    Simulates spectrum sensing by reading channel state files from Base Stations.
    Returns True if channel is BUSY, False if IDLE.
    """
    # Check for any channel state file in the current directory
    state_files = glob.glob("channel_state_BS*.json")
    
    for fname in state_files:
        try:
            with open(fname, "r") as f:
                data = json.load(f)
                # Only consider fresh data (within last 2 seconds)
                if time.time() - data.get("timestamp", 0) < 2.0:
                    if data.get("status") == "BUSY":
                        # If ANY base station says busy, the spectrum is busy
                        return True
        except Exception:
            pass
    return False

def send_message(sock, recipient, message_text):
    """Helper to construct and send a message.
       Control messages bypass sensing so handshakes succeed."""
    global chat_partner
    try:
        # 1. Spectrum Sensing (Listen Before Talk)
        # Only Secondary users need to sense. Primary users (JAZZ, etc.) just transmit.
        is_primary, op_name = get_node_type(NODE_ID)
        if NODE_ID and (not is_primary):
            # Control messages bypass sensing
            control_msgs = [CTL_CONNECT_REQUEST, CTL_CONNECT_ACCEPT, CTL_CONNECT_REJECT, CTL_DISCONNECT]
            if message_text not in control_msgs:
                print(f"[{NODE_ID}] Sensing spectrum...")
                if sense_channel_environment():
                    print(f"[{NODE_ID}] Channel is BUSY (Primary User Active). Initiating Backoff...")
                    # Simple Backoff Strategy: Wait and Retry
                    for i in range(3):
                        wait_time = 2.0
                        print(f"[{NODE_ID}] Waiting {wait_time}s (Attempt {i+1}/3)...")
                        time.sleep(wait_time)
                        if not sense_channel_environment():
                            print(f"[{NODE_ID}] Channel became IDLE. Proceeding.")
                            break
                    else:
                        print(f"[{NODE_ID}] Channel still BUSY. Transmitting anyway (Collision Risk!) or Dropping.")

        M = determine_M(message_text)
        nc, cp = pick_num_subcarriers_local(message_text, M)
        msg_bytes = message_text.encode("utf-8")
        energy, ofdm_sig = compute_ofdm_energy_from_message_bytes(msg_bytes, M, nc, cp)

        # Control messages bypass sensing
        control_msgs = [CTL_CONNECT_REQUEST, CTL_CONNECT_ACCEPT, CTL_CONNECT_REJECT, CTL_DISCONNECT]
        bypass_sensing = message_text in control_msgs

        # For data messages from the receiver (which may be secondary), we could optionally defer.
        # To keep behavior predictable, we will bypass sensing only for control messages.
        # Otherwise send immediately (the BS will enforce larger scale priority if needed).
        # (If you want receiver to behave as an opportunistic transmitter when it's not a primary,
        #  you can add additional checks here based on receiver ID in PRIMARY_SENDERS.)

        pkt = struct.pack(">H", len(message_text.encode())) + message_text.encode()
        plaintext = struct.pack(">H H B", M, nc, cp) + pkt
        enc = aes_gcm_encrypt(plaintext, KEY)
        enc_rs = rs.encode(enc)
        sensing_header = struct.pack(">d", float(energy))
        dst_b = recipient.encode("utf-8")
        wire_payload = struct.pack(">H", len(dst_b)) + dst_b + sensing_header + enc_rs
        sock.sendall(struct.pack(">I", len(wire_payload)) + wire_payload)

        if message_text not in control_msgs:
            record_message(recipient, message_text.encode("utf-8"), M, message_text)
        log_event(f"Sent to {recipient}: {message_text} (M={M}, NC={nc}, CP={cp}, energy={energy:.4e})")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send message: {e}")
        log_event(f"Send error: {e}")
        with state_lock:
            chat_partner = None
        return False

def receive_handler(sock):
    global chat_partner, pending_request_from
    sock.settimeout(2)
    while not stop_event.is_set():
        try:
            hdr = recv_full(sock, 4)
            if not hdr: continue
            length = struct.unpack(">I", hdr)[0]
            data = recv_full(sock, length)
            if not data: continue

            # Unpack sender ID, which is now prepended by the base station
            src_len = struct.unpack(">H", data[:2])[0]
            src_id = data[2:2+src_len].decode("utf-8")
            pos = 2 + src_len

            enc_blob_candidate = data[pos:]
            sensing_energy = None
            plaintext = None

            # Try path A: interpret first 8 bytes as sensing header (double) and decode the rest
            if len(enc_blob_candidate) >= 8:
                possible_hdr = enc_blob_candidate[:8]
                rest = enc_blob_candidate[8:]
                try:
                    sensing_energy_tmp = struct.unpack(">d", possible_hdr)[0]
                    # try RS decode + AES decrypt on rest
                    rs_decoded = rs.decode(rest)[0]
                    plaintext = aes_gcm_decrypt(rs_decoded, KEY)
                    sensing_energy = sensing_energy_tmp
                    enc_blob = rest
                except Exception:
                    plaintext = None

            # If path A failed, fall back to path B: treat the whole candidate as RS/AES payload
            if plaintext is None:
                try:
                    rs_decoded = rs.decode(enc_blob_candidate)[0]
                    plaintext = aes_gcm_decrypt(rs_decoded, KEY)
                    enc_blob = enc_blob_candidate
                    sensing_energy = None
                except Exception as e:
                    # Both attempts failed — log and skip this frame so we don't silently drop replies
                    continue

            # If we get here, plaintext is valid
            try:
                M, nc, cp = struct.unpack(">H H B", plaintext[:5])
                pkt = plaintext[5:]
                msg_len = struct.unpack(">H", pkt[:2])[0]
                msg = pkt[2:2+msg_len].decode("utf-8")
            except Exception:
                # malformed plaintext — skip
                continue

            # --- Handle Control Messages ---
            if msg == CTL_CONNECT_REQUEST:
                # MODIFICATION: Auto-accept or allow multiple connections
                with state_lock:
                    if src_id not in connected_peers:
                        print(f"\n[AUTO-ACCEPT] Incoming connection from {src_id}. Accepted.")
                        connected_peers.add(src_id)
                        # If we don't have a focus partner, set this one as default for replies
                        if not chat_partner:
                            chat_partner = src_id
                            print(f"[INFO] You are now replying to {chat_partner} by default.")
                    
                    # Send accept immediately
                    send_message(sock, src_id, CTL_CONNECT_ACCEPT)
                continue
            
            if msg == CTL_DISCONNECT:
                with state_lock:
                    if src_id in connected_peers:
                        connected_peers.remove(src_id)
                        print(f"\n[INFO] {src_id} disconnected.")
                    if src_id == chat_partner:
                        chat_partner = None
                        # Pick another partner if available
                        if connected_peers:
                            chat_partner = next(iter(connected_peers))
                            print(f"[INFO] Switched reply focus to {chat_partner}")
                continue

            # --- Handle Regular Messages ---
            with state_lock:
                # MODIFICATION: Accept message if sender is in our connected list (or just accept all)
                # We print it regardless of who we are currently typing to.
                print(f"\n<<< {src_id}: '{msg}'\n> ", end="", flush=True)
                record_message(src_id, msg.encode("utf-8"), M, msg)
                log_event(f"Received message from {src_id}: {msg} (M={M}, NC={nc}, CP={cp}, sensing_energy={sensing_energy})")
            
        except socket.timeout:
            continue
        except (ConnectionError, struct.error):
            if not stop_event.is_set():
                print("\n[INFO] Connection lost. Press Enter to exit.")
            stop_event.set()
            break
        except Exception:
            # Catch-all to keep the receive loop alive on unexpected errors
            continue

def main_loop(sock):
    global chat_partner, pending_request_from
    while not stop_event.is_set():
        try:
            # Determine prompt based on current state
            with state_lock:
                current_partner = chat_partner
                # We no longer block on pending requests since we auto-accept
            
            if current_partner:
                prompt = f"To {current_partner}> "
            else:
                prompt = "Waiting for connections... (type 'exit' to quit): "
            
            user_input = input(prompt).strip()

            if not user_input:
                continue

            if user_input.lower() == 'exit':
                # Disconnect from everyone
                with state_lock:
                    for peer in list(connected_peers):
                        send_message(sock, peer, CTL_DISCONNECT)
                stop_event.set()
                break
            
            # Switch focus command (optional helper)
            if user_input.lower().startswith("/switch "):
                target = user_input.split(" ")[1].upper()
                with state_lock:
                    if target in connected_peers:
                        chat_partner = target
                        print(f"[INFO] Now replying to {target}")
                    else:
                        print(f"[ERROR] {target} is not connected.")
                continue

            # State: In an active chat
            if current_partner:
                if not send_message(sock, current_partner, user_input):
                    print("[ERROR] Connection lost or send failed.")
            else:
                print("[INFO] No active chat partner to reply to. Wait for connection.")


        except (EOFError, KeyboardInterrupt):
            stop_event.set()
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            log_event(f"Error in main loop: {e}")
            stop_event.set()


# ---------- Plot generation & export ----------
def compute_ber_for_snr(tx_bits, M, snr_db):
    """Simulate AWGN on tx symbols and compute bit errors vs SNR (simple link-level sim)."""
    if len(tx_bits) == 0: return 1.0
    symbols = qam_mod(tx_bits, M)
    snr_linear = 10**(snr_db/10.0)
    noise_var = 1.0 / (2 * snr_linear)
    noise = (np.sqrt(noise_var) * (np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape)))
    rx = symbols + noise
    sym_err = np.mean(np.any(np.abs(rx - symbols) > 1e-6, axis=0)) if symbols.size>0 else 1.0
    k = int(np.log2(M))
    return min(1.0, sym_err * k * 0.5)

def export_all_results(node_id):
    figs = []
    for peer, entries in messages_by_peer.items():
        for i, e in enumerate(entries):
            tx_syms, ofdm_sig = e.get("tx_symbols", np.array([])), e.get("ofdm_sig", np.array([]))
            message_text, M, nc, cp = e.get("text", "Unknown message"), e.get("M", 16), e.get("nc", 64), e.get("cp", 8)
            
            if tx_syms.size > 0:
                figs.append(make_constellation_plot(tx_syms, f"{peer} - Constellation Diagram ({i})", message_text, M, nc, cp))
                figs.append(make_ofdm_plot(ofdm_sig, f"{peer} - OFDM I/Q Signal ({i})", message_text, M, nc, cp))
                # Sensing visualization based on the OFDM computed for the message (uses message bits)
                threshold = np.mean(np.abs(ofdm_sig)**2) * 0.6 if ofdm_sig.size>0 else None
                figs.append(make_sensing_plot(ofdm_sig, f"{peer} - Spectrum Sensing ({i})", message_text, threshold))

    if messages_by_peer:
        peer, entry = next(iter(messages_by_peer.items()))
        M, bits, message_text = entry[0].get("M", 16), bits_from_bytes(entry[0].get("bytes", b"")), entry[0].get("text", "Sample message")
        
        snrs = np.arange(0, 21, 2)
        bers = [compute_ber_for_snr(bits, M, s) for s in snrs]
        
        snr_fig = plt.figure(figsize=(8,6))
        plt.semilogy(snrs, bers, marker='o', linewidth=2, markersize=8)
        plt.title("SNR vs BER Performance"); plt.xlabel("SNR (dB)"); plt.ylabel("Bit Error Rate (BER)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        textstr = f'Message: "{message_text[:50]}{"..." if len(message_text)>50 else ""}"\nModulation: QAM-{M}\nBER@10dB: {bers[5]:.2e}\nBER@20dB: {bers[10]:.2e}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        plt.figtext(0.02, 0.02, textstr, fontsize=10, bbox=props)
        figs.append(snr_fig)
    else:
        empty_fig = plt.figure(figsize=(6,4)); plt.text(0.5, 0.5, "No data received", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("No Messages Received"); figs.append(empty_fig)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pdf_path = os.path.join(DATA_DIR, f"{node_id}_plots_{timestamp}.pdf")
    save_plots_to_pdf(pdf_path, figs)

    log_path = os.path.join(DATA_DIR, f"{node_id}_log_{timestamp}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"5G Communication Pipeline - Receiver Node {node_id}\n"); f.write(f"Generated: {datetime.datetime.utcnow().isoformat()}Z\n"); f.write("="*50 + "\n\n")
        for ev in base_station_events: f.write(ev + "\n")
        f.write("\n--- Messages Log ---\n")
        for peer, entries in messages_by_peer.items():
            f.write(f"Peer: {peer}\n")
            for e in entries:
                f.write(f"Time: {datetime.datetime.utcfromtimestamp(e['ts']).isoformat()}\n")
                f.write(f"Length: {len(e.get('bytes',b''))} bytes | Modulation: QAM-{e.get('M', 'N/A')}\n")
                f.write(f"Subcarriers: {e.get('nc', 'N/A')} | CP: {e.get('cp', 'N/A')}\n")
                f.write(f"Message: {e.get('text', 'Unknown')}\n"); f.write("-" * 40 + "\n")
    
    print(f"[EXPORT] Saved plots to {pdf_path} and logs to {log_path}")
    log_event(f"Exported plots to {pdf_path} and logs to {log_path}")

# ---------- Main ----------
def main():
    global NODE_ID
    node_id = ""
    while not re.match(r"^R([1-9]|[1-5][0-9]|60)$", node_id):
        node_id = input("Enter your Receiver ID (R1-R60): ").strip().upper()
    
    NODE_ID = node_id
    cleanup_old_files(node_id)

    node_num = int(node_id[1:])
    BS_HOST, BS_PORT = (BS1_HOST, BS1_PORT) if 1 <= node_num <= 30 else (BS2_HOST, BS2_PORT)
    node_pos = (random.uniform(10, 140), random.uniform(-50, 50)) if 1 <= node_num <= 30 else (random.uniform(210, 340), random.uniform(-50, 50))

    try:
        sock = socket.create_connection((BS_HOST, BS_PORT), timeout=5)
        reg_msg = {"type": "receiver", "id": node_id, "pos": list(node_pos)}
        sock.sendall((json.dumps(reg_msg) + "\n").encode("utf-8"))
        print(f"RECEIVER '{node_id}' registered with Base Station at {BS_HOST}:{BS_PORT}.")
        log_event(f"Registered {node_id} to {BS_HOST}:{BS_PORT}")
    except Exception as e:
        print(f"Failed to connect to Base Station: {e}")
        return

    threading.Thread(target=receive_handler, args=(sock,), daemon=True).start()
    try:
        main_loop(sock)
    finally:
        export_all_results(node_id)
        sock.close()
        print(f"\nRECEIVER '{node_id}': Exiting.")

if __name__ == "__main__":
    main()
