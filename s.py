# Dynamic_Sender.py (Enhanced + Agent) - updated with Primary/Secondary and energy-based sensing
# - Manual sender CLI + optional autonomous agent
# - Records PHY data, creates constellation/OFDM/IQ plots
# - SNR vs BER simulation, PDF + TXT export on exit
# - Agent can auto-transmit to DEFAULT_RECIPIENT based on NODE_POS/logic
# - Energy-based spectrum sensing done locally on message bits (uses message bits not random bits)
# - Shows sensing results in exported plots

import socket, struct, threading, json, hashlib, random, re, os, time, datetime
from reedsolo import RSCodec
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob

# ---------- CONFIG: Base Stations ----------
BS1_HOST, BS1_PORT = "127.0.0.1", 50050
BS2_HOST, BS2_PORT = "127.0.0.1", 50051

# ---------- Primary / Secondary setup ----------
# Primary operators get strict priority. Others are opportunistic.
PRIMARY_SENDERS = ["JAZZ", "UFONE", "TELENOR", "WARID", "STARLINK", "ZONG", "SCO", "PTCL"]
# Treat any sender ID not in PRIMARY_SENDERS as secondary (opportunistic)

# ---------- AGENT / NODE CONFIG (change if needed) ----------
NODE_ID = "JAZZ"                      # default ID used when running agent mode
NODE_POS = (220.0, 0.0)             # placed in range of BS2 in your description
DEFAULT_RECIPIENT = "R1"            # recipient the agent targets automatically
AGENT_ENABLED = False               # Changed to False by default to prevent continuous loop
AGENT_INTERVAL = 15.0               # Increased interval to 15 seconds
AGENT_PAYLOADS = ["Hello from agent", "Telemetry sample", "ping"]  # rotate payloads

# ---------- Crypto / FEC ----------
PASSPHRASE = b"very secret passphrase - change this!"
KEY = hashlib.sha256(PASSPHRASE).digest()
rs = RSCodec(40)
stop_event = threading.Event()
plot_lock = threading.Lock()

# --- Session Management ---
# State variables for managing chat sessions
chat_partner = None
connection_status = threading.Event() # Used to wait for accept/reject
connection_accepted = False
state_lock = threading.Lock()

# Control messages for session management
CTL_CONNECT_REQUEST = "__CONNECT_REQUEST__"
CTL_CONNECT_ACCEPT = "__CONNECT_ACCEPT__"
CTL_CONNECT_REJECT = "__CONNECT_REJECT__"
CTL_DISCONNECT = "__DISCONNECT__"
# sensing control (kept for future extension)
CTL_SENSE_QUERY = "__SENSE_QUERY__"
CTL_SENSE_REPLY = "__SENSE_REPLY__"

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

# ---------- PHY helpers ----------
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

# ---------- Agent-local decision helpers (from your snippet) ----------
def determine_M_local(msg_or_path: str) -> int:
    if os.path.isfile(msg_or_path):
        ext = os.path.splitext(msg_or_path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            return 256
        if ext in [".wav", ".mp3"]:
            return 64
    n = len(msg_or_path)
    if n <= 100: return 16
    if n <= 500: return 64
    return 256

def pick_num_subcarriers_local(encoded_bytes_or_text, M):
    candidates = [16, 32, 64, 128]
    try:
        if isinstance(encoded_bytes_or_text, (bytes, bytearray)):
            bits = bits_from_bytes(encoded_bytes_or_text)
        else:
            bits = bits_from_bytes(str(encoded_bytes_or_text).encode('utf-8'))
        tx_symbols = qam_mod(bits, M)
    except Exception:
        tx_symbols = qam_mod(np.zeros(2, dtype=np.uint8), 16)
    lat, mob = [], []
    for nc in candidates:
        sig = ofdm_mod(tx_symbols, nc, max(4, nc//8))
        lat.append(len(sig)); mob.append(1.0/nc)
    lat = np.array(lat); mob = np.array(mob)
    lat_n = (lat - lat.min()) / (np.ptp(lat) + 1e-12)
    mob_n = (mob - mob.min()) / (np.ptp(mob) + 1e-12)
    cost = 0.7 * lat_n + 0.3 * mob_n
    idx = int(np.argmin(cost))
    nc = candidates[idx]; cp = max(4, nc//8)
    return nc, cp

# ---------- Data storage & logging ----------
DATA_DIR = "node_logs"
os.makedirs(DATA_DIR, exist_ok=True)
messages_sent = []
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

def record_message(recipient, text_bytes, M, message_text):
    bits = bits_from_bytes(text_bytes)
    nc, cp = pick_num_subcarriers_local(text_bytes, M)
    tx_syms = qam_mod(bits, M)
    ofdm_sig = ofdm_mod(tx_syms, nc, cp)
    messages_sent.append({
        "recipient": recipient,
        "bytes": text_bytes,
        "text": message_text,
        "M": M,
        "nc": nc,
        "cp": cp,
        "tx_syms": tx_syms,
        "ofdm": ofdm_sig,
        "ts": time.time()
    })
    print(f"[PHY_PARAMS] M={M}, Subcarriers={nc}, CP={cp} for message: '{message_text[:30]}...'")

# ---------- SNR vs BER sim ----------
def compute_ber_for_snr(tx_bits, M, snr_db):
    if len(tx_bits) == 0:
        return 1.0
    symbols = qam_mod(tx_bits, M)
    snr_linear = 10**(snr_db/10.0)
    noise_var = 1.0 / (2 * snr_linear)
    noise = (np.sqrt(noise_var) * (np.random.randn(*symbols.shape) + 1j*np.random.randn(*symbols.shape)))
    rx = symbols + noise
    sym_err = np.mean(np.abs(rx - symbols) > 0.5) if symbols.size>0 else 1.0
    k = int(np.log2(M))
    return min(1.0, sym_err * k * 0.5)

# ---------- Plot export ----------
def export_all_results(node_id):
    figs = []
    for i, msg in enumerate(messages_sent):
        tx_syms, ofdm_sig = msg["tx_syms"], msg["ofdm"]
        message_text = msg.get("text", "Unknown message")
        M, nc, cp = msg["M"], msg["nc"], msg["cp"]
        
        figs.append(make_constellation_plot(tx_syms, f"{msg['recipient']} - Constellation Diagram ({i})", 
                                          message_text, M, nc, cp))
        figs.append(make_ofdm_plot(ofdm_sig, f"{msg['recipient']} - OFDM I/Q Signal ({i})", 
                                 message_text, M, nc, cp))
        # Spectrum sensing plot for this message (uses actual message OFDM samples)
        # Threshold chosen as small multiple of mean noise estimate (simple heuristic)
        threshold = np.mean(np.abs(ofdm_sig)**2) * 0.6 if ofdm_sig.size>0 else None
        figs.append(make_sensing_plot(ofdm_sig, f"{msg['recipient']} - Spectrum Sensing ({i})", message_text, threshold))
    
    if messages_sent:
        sample_msg = messages_sent[0]
        M = sample_msg["M"]
        bits = bits_from_bytes(sample_msg["bytes"])
        message_text = sample_msg.get("text", "Sample message")
        
        snrs = np.arange(0, 21, 2)
        bers = [compute_ber_for_snr(bits, M, s) for s in snrs]
        
        snr_fig = plt.figure(figsize=(8,6))
        plt.semilogy(snrs, bers, marker='o', linewidth=2, markersize=8)
        plt.title("SNR vs BER Performance")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate (BER)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        textstr = f'Message: "{message_text[:50]}{"..." if len(message_text)>50 else ""}"\nModulation: QAM-{M}\nBER@10dB: {bers[5]:.2e}\nBER@20dB: {bers[10]:.2e}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        plt.figtext(0.02, 0.02, textstr, fontsize=10, bbox=props)
        
        figs.append(snr_fig)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pdf_path = os.path.join(DATA_DIR, f"{node_id}_plots_{timestamp}.pdf")
    save_plots_to_pdf(pdf_path, figs)
    
    log_path = os.path.join(DATA_DIR, f"{node_id}_log_{timestamp}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"5G Communication Pipeline - Sender Node {node_id}\n")
        f.write(f"Generated: {datetime.datetime.utcnow().isoformat()}Z\n")
        f.write("="*50 + "\n\n")
        
        for ev in base_station_events:
            f.write(ev + "\n")
        
        f.write("\n--- Messages Sent ---\n")
        for m in messages_sent:
            f.write(f"Time: {datetime.datetime.utcfromtimestamp(m['ts']).isoformat()}\n")
            f.write(f"To: {m['recipient']} | Length: {len(m['bytes'])} bytes\n")
            f.write(f"Modulation: QAM-{m['M']} | Subcarriers: {m['nc']} | CP: {m['cp']}\n")
            f.write(f"Message: {m.get('text', 'Unknown')}\n")
            f.write("-" * 40 + "\n")
    
    print(f"[EXPORT] Saved plots to {pdf_path} and logs to {log_path}")
    log_event(f"Exported plots to {pdf_path} and logs to {log_path}")

# ---------- Sensing helpers ----------
def compute_ofdm_energy_from_message_bytes(message_bytes, M, nc, cp):
    """Build OFDM from message bytes and compute average energy (power)."""
    bits = bits_from_bytes(message_bytes)
    tx_syms = qam_mod(bits, M)
    ofdm_sig = ofdm_mod(tx_syms, nc, cp)
    energy = np.mean(np.abs(ofdm_sig)**2) if ofdm_sig.size>0 else 0.0
    return energy, ofdm_sig

# ---------- Networking handlers ----------
def send_message(sock, recipient, message_text):
    """Helper to construct and send a message.
       For opportunistic (secondary) senders we perform local energy-based sensing
       using the message bits. Primary senders always send immediately."""
    global chat_partner
    try:
        M = determine_M(message_text)
        nc, cp = pick_num_subcarriers_local(message_text, M)

        # SENSING: compute OFDM energy from message bits (not random bits)
        msg_bytes = message_text.encode("utf-8")
        energy, ofdm_sig = compute_ofdm_energy_from_message_bytes(msg_bytes, M, nc, cp)

        # Determine primary/secondary by NODE_ID presence in PRIMARY_SENDERS
        is_primary = NODE_ID.upper() in PRIMARY_SENDERS

        # Control messages bypass sensing (always send)
        control_msgs = [CTL_CONNECT_REQUEST, CTL_CONNECT_ACCEPT, CTL_CONNECT_REJECT, CTL_DISCONNECT]
        bypass_sensing = (message_text in control_msgs) or is_primary

        # If secondary and not bypassing, decide whether to defer
        if not bypass_sensing:
            # Heuristic threshold: a small multiple of mean power of the message OFDM
            threshold = np.mean(np.abs(ofdm_sig)**2) * 0.6 if ofdm_sig.size>0 else 0.0
            # If energy significantly above threshold => assume busy and defer (opportunistic)
            if energy > threshold and threshold > 0:
                print(f"[SENSING] Secondary node {NODE_ID} deferring transmission (E={energy:.3e} > thr={threshold:.3e})")
                log_event(f"Deferred send to {recipient}: E={energy:.3e} thr={threshold:.3e}")
                return False  # indicate not sent
        # Compose plaintext and include a small sensing metadata header (non-invasive): M,nc,cp + pkt
        pkt = struct.pack(">H", len(message_text.encode())) + message_text.encode()
        plaintext = struct.pack(">H H B", M, nc, cp) + pkt
        enc = aes_gcm_encrypt(plaintext, KEY)
        enc_rs = rs.encode(enc)

        # Attach a tiny sensing header (double) representing energy computed from message bits.
        dst_b = recipient.encode("utf-8")
        sensing_header = struct.pack(">d", float(energy))
        # Keep protocol backwards-compatible by placing sensing header after dst length+dst id.
        wire_payload = struct.pack(">H", len(dst_b)) + dst_b + sensing_header + enc_rs

        sock.sendall(struct.pack(">I", len(wire_payload)) + wire_payload)

        # Log and record the message (we store ofdm_sig we computed so we can plot sensing)
        if message_text not in control_msgs:
             record_message(recipient, msg_bytes, M, message_text)
        log_event(f"Sent to {recipient}: {message_text} (M={M}, NC={nc}, CP={cp}, energy={energy:.4e}, primary={is_primary})")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send message: {e}")
        log_event(f"Send error: {e}")
        with state_lock:
            chat_partner = None
        return False


def receive_handler(sock):
    global chat_partner, connection_accepted
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
            # Next may include sensing header (8 bytes) if produced by sender
            pos = 2 + src_len
            # If remaining length >= 8 and next bytes look like a double, we parse and forward
            sensing_header = None
            if len(data[pos:]) >= 8:
                try:
                    sensing_header = struct.unpack(">d", data[pos:pos+8])[0]
                    pos += 8
                except Exception:
                    sensing_header = None

            enc_blob = data[pos:]

            rs_decoded = rs.decode(enc_blob)[0]
            plaintext = aes_gcm_decrypt(rs_decoded, KEY)
            pkt = plaintext[5:] # Skip M, NC, CP
            msg_len = struct.unpack(">H", pkt[:2])[0]
            msg = pkt[2:2+msg_len].decode("utf-8")

            # --- Handle Control Messages ---
            if msg == CTL_CONNECT_ACCEPT:
                with state_lock:
                    connection_accepted = True
                connection_status.set()
                continue
            
            if msg == CTL_CONNECT_REJECT:
                with state_lock:
                    connection_accepted = False
                connection_status.set()
                continue

            if msg == CTL_DISCONNECT:
                 with state_lock:
                    if src_id == chat_partner:
                        print(f"\n[INFO] {chat_partner} has disconnected. Press Enter to continue.", flush=True)
                        chat_partner = None
                 continue

            # --- Handle Regular Messages ---
            with state_lock:
                if src_id == chat_partner:
                    print(f"\n<<< {src_id}: '{msg}'\n> ", end="", flush=True)
                    log_event(f"Received reply from {src_id}: {msg}")

        except socket.timeout:
            continue
        except Exception:
            continue

# ---------- Agent thread ----------
def agent_thread(sock):
    """Autonomous agent that periodically sends payloads to DEFAULT_RECIPIENT."""
    idx = 0
    while AGENT_ENABLED and not stop_event.is_set():
        try:
            # Agent only runs if not in an active chat
            with state_lock:
                is_in_chat = chat_partner is not None
            
            if not is_in_chat:
                payload = AGENT_PAYLOADS[idx % len(AGENT_PAYLOADS)]
                idx += 1
                print(f"[AGENT] Sending to {DEFAULT_RECIPIENT}: '{payload}'")
                send_message(sock, DEFAULT_RECIPIENT, payload)
            
            # Wait for the interval, checking for stop event periodically
            for _ in range(int(AGENT_INTERVAL * 2)):
                if stop_event.is_set(): break
                time.sleep(0.5)
                
        except Exception as e:
            if not stop_event.is_set():
                log_event(f"Agent error: {e}")
                print(f"[AGENT] Error: {e}")
            break

# ---------- Interactive send handler ----------
def send_handler(sock, node_id):
    global chat_partner, connection_accepted

    if AGENT_ENABLED:
        threading.Thread(target=agent_thread, args=(sock,), daemon=True).start()
        print(f"[AGENT] Autonomous agent started. Will send to {DEFAULT_RECIPIENT} when idle.")

    while not stop_event.is_set():
        with state_lock:
            in_chat = chat_partner is not None
        
        if in_chat:
            # --- In an active chat session ---
            message = input(f"> ")
            if message.strip().lower() == 'exit':
                send_message(sock, chat_partner, CTL_DISCONNECT)
                print(f"[INFO] Disconnected from {chat_partner}.")
                with state_lock:
                    chat_partner = None
                continue
            
            if not send_message(sock, chat_partner, message):
                print("[ERROR] Connection lost or message deferred.")
        
        else:
            # --- Not in a chat, waiting to connect ---
            print("\n" + "="*50)
            recipient = input("Enter recipient ID to connect (or 'exit'): ").strip().upper()
            print("="*50)

            if not recipient:
                continue
            
            if recipient == 'EXIT':
                stop_event.set()
                break

            if not re.match(r"^[RS]([1-9]|[1-5][0-9]|60)$", recipient):
                print(f"[ERROR] Invalid ID: {recipient}. Use format R1-R60 or S1-S60.")
                continue

            # Send connection request and wait for a response
            send_message(sock, recipient, CTL_CONNECT_REQUEST)
            print(f"[INFO] Sent connection request to {recipient}. Waiting for reply...")
            
            connection_status.clear()
            # Wait for 30 seconds for the other user to accept
            responded = connection_status.wait(timeout=30.0)

            if not responded:
                print(f"[INFO] No response from {recipient}.")
                continue
            
            with state_lock:
                accepted = connection_accepted

            if accepted:
                with state_lock:
                    chat_partner = recipient
                print(f"\n[SUCCESS] Connection with {chat_partner} established! You can now chat.")
                print("Type 'exit' to end the session.")
                print(f"> ", end="") # Initial prompt
            else:
                print(f"[INFO] {recipient} rejected the connection request.")

# ---------- Main ----------
def main():
    node_id = NODE_ID if AGENT_ENABLED and NODE_ID else ""
    if not node_id:
        while not re.match(r"^S([1-9]|[1-5][0-9]|60)$", node_id):
            node_id = input("Enter your Sender ID (S1-S60): ").strip().upper()
    
    cleanup_old_files(node_id)
    
    node_num = int(node_id[1:])
    if 1 <= node_num <= 30:
        BS_HOST, BS_PORT = BS1_HOST, BS1_PORT
        node_pos = (random.uniform(10, 140), random.uniform(-50, 50))
    else:
        BS_HOST, BS_PORT = BS2_HOST, BS2_PORT
        node_pos = NODE_POS if AGENT_ENABLED else (random.uniform(210, 340), random.uniform(-50, 50))

    try:
        sock = socket.create_connection((BS_HOST, BS_PORT), timeout=5)
        reg_msg = {"type": "sender", "id": node_id, "pos": list(node_pos)}
        sock.sendall((json.dumps(reg_msg) + "\n").encode("utf-8"))
        print(f"SENDER '{node_id}' registered with Base Station at {BS_HOST}:{BS_PORT}.")
        log_event(f"Registered {node_id} to {BS_HOST}:{BS_PORT}")
    except Exception as e:
        print(f"Failed to connect to Base Station: {e}")
        return

    threading.Thread(target=receive_handler, args=(sock,), daemon=True).start()
    try:
        send_handler(sock, node_id)
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Shutting down...")
    finally:
        with state_lock:
            if chat_partner:
                send_message(sock, chat_partner, CTL_DISCONNECT)
        stop_event.set()
        export_all_results(node_id)
        sock.close()
        print(f"\nSENDER '{node_id}': Exiting.")

if __name__ == "__main__":
    main()
