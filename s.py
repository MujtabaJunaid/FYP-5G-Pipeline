# Dynamic_Sender.py (s.py) - FIXED PLOTTING & EXPORT
# - Added matplotlib.use('Agg') to prevent backend crashes
# - Restored detailed Plot Metadata (Text boxes)
# - Thread-safe export to prevent corrupt PDF files

print("\n[DEBUG] SENDER: LOADING FIXED PLOTTING MODULE...\n")

import socket, struct, threading, json, hashlib, random, re, os, time, datetime
from reedsolo import RSCodec
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
import numpy as np

# --- PLOTTING SETUP (CRITICAL FIX) ---
import matplotlib
matplotlib.use("Agg") # Force headless backend to prevent corruption
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob

# --- SIMULATED SPECTRUM ENVIRONMENT ---
ENVIRONMENTAL_NOISE_FLOOR = 1.0e-10 
ENVIRONMENTAL_THRESHOLD = ENVIRONMENTAL_NOISE_FLOOR * 2.0
ENV_STATE = 0 
ENV_TRANSITION_MATRIX = [[0.90, 0.10], [0.30, 0.70]]

BS1_HOST, BS1_PORT = "127.0.0.1", 50050
BS2_HOST, BS2_PORT = "127.0.0.1", 50051

PRIMARY_SENDERS = ["JAZZ", "UFONE", "TELENOR", "WARID", "STARLINK", "ZONG", "SCO", "PTCL"]

# ---------- Crypto ----------
PASSPHRASE = b"very secret passphrase - change this!"
SALT = b'\x00'*16 
KEY = PBKDF2(PASSPHRASE, SALT, dkLen=32, count=100000)
rs = RSCodec(40)
stop_event = threading.Event()
plot_lock = threading.Lock()
log_lock = threading.Lock() # Protect message logs

# ---------- AGENT ----------
NODE_ID = "JAZZ"; NODE_POS = (220.0, 0.0); DEFAULT_RECIPIENT = "R1"
AGENT_ENABLED = False; AGENT_INTERVAL = 15.0; AGENT_PAYLOADS = ["Hello from agent", "Telemetry sample", "ping"]

# --- Session ---
chat_partner = None
connection_status = threading.Event()
connection_accepted = False
state_lock = threading.Lock()

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

def determine_M(msg): return 256 if len(msg)>500 else (64 if len(msg)>100 else 16)

def recv_full(sock, length):
    data = b""
    while len(data) < length:
        try:
            more = sock.recv(length - len(data))
            if not more: raise ConnectionError("socket closed")
            data += more
        except socket.timeout: return None
    return data

# ---------- PHY ----------
def bits_from_bytes(b): return np.unpackbits(np.frombuffer(b, dtype=np.uint8))
def qam_mod(bits, M):
    k = int(np.log2(M))
    if len(bits)%k!=0: bits = np.concatenate([bits, np.zeros(k-(len(bits)%k), dtype=np.uint8)])
    ints = bits.reshape((-1,k)).dot(1<<np.arange(k-1,-1,-1))
    scale = np.sqrt((2.0/3.0)*(M-1)) if M>1 else 1.0
    return ((2*(ints%int(np.sqrt(M)))-(int(np.sqrt(M))-1))/scale) + 1j*((2*(ints//int(np.sqrt(M)))-(int(np.sqrt(M))-1))/scale)
def ofdm_mod(syms, nc, cp):
    if len(syms)==0: return np.array([])
    n = int(np.ceil(len(syms)/nc))
    padded = np.pad(syms, (0, n*nc-len(syms)))
    ifft = np.fft.ifft(padded.reshape((n, nc)), axis=1)
    return np.hstack([ifft[:, -cp:], ifft]).flatten()

def sense_environment(sock):
    global ENV_STATE
    if random.random() < ENV_TRANSITION_MATRIX[ENV_STATE][1-ENV_STATE]: ENV_STATE = 1 - ENV_STATE
    noise = ENVIRONMENTAL_NOISE_FLOOR * np.random.uniform(0.8, 1.2)
    if ENV_STATE == 1: noise += ENVIRONMENTAL_NOISE_FLOOR * 10.0
    return noise

# ---------- PLOTTING (Robust) ----------
def make_constellation_plot(symbols, title, message_text, M, nc, cp):
    fig = plt.figure(figsize=(8,6))
    if len(symbols) > 0:
        plt.scatter(np.real(symbols), np.imag(symbols), s=6)
    plt.title(title); plt.grid(True); plt.xlabel("I"); plt.ylabel("Q")
    
    # Metadata text box
    textstr = f'Msg: "{message_text[:30]}..."\nMod: QAM-{M}\nNC: {nc}, CP: {cp}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=props)
    return fig

def make_ofdm_plot(signal, title, message_text, M, nc, cp):
    fig = plt.figure(figsize=(8,4))
    if len(signal) > 0:
        # Plot only first 300 samples to avoid clutter
        limit = min(300, len(signal))
        plt.plot(np.real(signal[:limit]), label="I"); plt.plot(np.imag(signal[:limit]), label="Q")
    plt.title(title); plt.legend(); plt.grid(True)
    return fig

def make_sensing_plot(ofdm_sig, title, message_text, threshold=None):
    fig = plt.figure(figsize=(8,3))
    power = np.abs(ofdm_sig)**2 if ofdm_sig.size>0 else np.array([])
    if power.size>0:
        limit = min(1000, len(power))
        plt.plot(power[:limit])
        if threshold: plt.hlines(threshold, 0, limit, linestyles='--', color='r')
    plt.title(title); plt.grid(True); plt.ylabel("Power")
    return fig

def save_plots_to_pdf(pdf_path, figs):
    with plot_lock:
        try:
            with PdfPages(pdf_path) as pdf:
                for f in figs: 
                    pdf.savefig(f)
                    plt.close(f)
        except Exception as e:
            print(f"[ERROR] Plot saving failed: {e}")

# ---------- Logging ----------
DATA_DIR = "node_logs"
os.makedirs(DATA_DIR, exist_ok=True)
messages_sent = []; base_station_events = []

def log_event(txt): base_station_events.append(f"{datetime.datetime.utcnow().isoformat()}Z {txt}")
def cleanup_old_files(nid):
    for f in glob.glob(os.path.join(DATA_DIR, f"{nid}_*")):
        try: os.remove(f)
        except: pass

def record_message(recip, b, M, txt):
    bits = bits_from_bytes(b); nc=64; cp=8
    tx = qam_mod(bits, M); sig = ofdm_mod(tx, nc, cp)
    with log_lock:
        messages_sent.append({"recipient": recip, "bytes": b, "text": txt, "M": M, "nc": nc, "cp": cp, "tx_syms": tx, "ofdm": sig, "ts": time.time()})

def export_all_results(node_id):
    print(f"\n[EXPORT] Generating plots for {len(messages_sent)} messages...")
    figs = []
    # Thread-safe copy
    with log_lock:
        msgs_copy = list(messages_sent)
        
    for i, m in enumerate(msgs_copy):
        try:
            figs.append(make_constellation_plot(m['tx_syms'], f"{m['recipient']} Constellation ({i})", m['text'], m['M'], m['nc'], m['cp']))
            figs.append(make_ofdm_plot(m['ofdm'], f"{m['recipient']} OFDM ({i})", m['text'], m['M'], m['nc'], m['cp']))
            thresh = np.mean(np.abs(m['ofdm'])**2) * 0.6 if len(m['ofdm']) > 0 else 0
            figs.append(make_sensing_plot(m['ofdm'], f"{m['recipient']} Sensing ({i})", m['text'], thresh))
        except Exception as e:
            print(f"[WARN] Failed to plot message {i}: {e}")

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pdf_path = os.path.join(DATA_DIR, f"{node_id}_plots_{timestamp}.pdf")
    save_plots_to_pdf(pdf_path, figs)
    
    with open(os.path.join(DATA_DIR, f"{node_id}_log_{timestamp}.txt"), "w") as f:
        for ev in base_station_events: f.write(ev + "\n")
        for m in msgs_copy: f.write(f"To {m['recipient']}: {m['text']}\n")
    print(f"[EXPORT] Saved plots to {pdf_path}")

def compute_ofdm_energy_from_message_bytes(message_bytes, M, nc, cp):
    bits = bits_from_bytes(message_bytes)
    tx_syms = qam_mod(bits, M)
    ofdm_sig = ofdm_mod(tx_syms, nc, cp)
    return (np.mean(np.abs(ofdm_sig)**2) if ofdm_sig.size>0 else 0.0), ofdm_sig

# ---------- Networking ----------
def send_message(sock, recipient, message_text):
    try:
        M = determine_M(message_text); nc, cp = 64, 8
        if (message_text not in [CTL_CONNECT_REQUEST, CTL_CONNECT_ACCEPT, CTL_CONNECT_REJECT, CTL_DISCONNECT]) and (NODE_ID not in PRIMARY_SENDERS):
            if sense_environment(sock) > ENVIRONMENTAL_THRESHOLD:
                print(f"[SENSING] Busy. Backing off..."); return False

        msg_bytes = message_text.encode("utf-8")
        energy, _ = compute_ofdm_energy_from_message_bytes(msg_bytes, M, nc, cp)

        pkt = struct.pack(">H", len(msg_bytes)) + msg_bytes
        plaintext = struct.pack(">H H B", M, nc, cp) + pkt
        enc = aes_gcm_encrypt(plaintext, KEY)
        enc_rs = rs.encode(enc)
        
        dst_b = recipient.encode("utf-8")
        payload = struct.pack(">H", len(dst_b)) + dst_b + struct.pack(">d", float(energy)) + enc_rs
        sock.sendall(struct.pack(">I", len(payload)) + payload)

        if message_text not in [CTL_CONNECT_REQUEST, CTL_CONNECT_ACCEPT, CTL_CONNECT_REJECT, CTL_DISCONNECT]:
            record_message(recipient, msg_bytes, M, message_text)
        return True
    except: return False

def receive_handler(sock):
    global chat_partner, connection_accepted
    sock.settimeout(2)
    while not stop_event.is_set():
        try:
            hdr = recv_full(sock, 4)
            if not hdr: continue
            length = struct.unpack(">I", hdr)[0]
            data = recv_full(sock, length)
            
            src_len = struct.unpack(">H", data[:2])[0]
            src_id = data[2:2+src_len].decode("utf-8")
            pos = 2 + src_len + 8 # Skip ID + Sensing Header
            
            try: plaintext = aes_gcm_decrypt(rs.decode(data[pos:])[0], KEY)
            except: continue
            
            pkt = plaintext[5:]; msg_len = struct.unpack(">H", pkt[:2])[0]
            msg = pkt[2:2+msg_len].decode("utf-8")

            if msg == CTL_CONNECT_ACCEPT: connection_status.set(); connection_accepted = True; continue
            if msg == CTL_CONNECT_REJECT: connection_status.set(); connection_accepted = False; continue
            if msg == CTL_DISCONNECT:
                with state_lock:
                    if src_id == chat_partner: print(f"\n[INFO] {chat_partner} disconnected.\n> ", end="", flush=True); chat_partner = None
                continue

            with state_lock:
                if src_id == chat_partner: print(f"\n<<< {src_id}: '{msg}'\n> ", end="", flush=True)

        except socket.timeout: continue
        except: continue

def send_handler(sock, node_id):
    global chat_partner
    if AGENT_ENABLED: threading.Thread(target=agent_thread, args=(sock,), daemon=True).start()
    while not stop_event.is_set():
        with state_lock: in_chat = chat_partner is not None
        if in_chat:
            message = input(f"> ")
            if message.strip().lower() == 'exit':
                send_message(sock, chat_partner, CTL_DISCONNECT); 
                with state_lock: chat_partner = None; continue
            send_message(sock, chat_partner, message)
        else:
            print("\n" + "="*30)
            recipient = input("Enter recipient ID (or 'exit'): ").strip().upper()
            if recipient == 'EXIT': stop_event.set(); break
            
            send_message(sock, recipient, CTL_CONNECT_REQUEST)
            print(f"[INFO] Connecting to {recipient}...")
            connection_status.clear()
            if connection_status.wait(timeout=60.0):
                if connection_accepted:
                    with state_lock: chat_partner = recipient
                    print(f"[SUCCESS] Connected to {recipient}!")
                else: print("[INFO] Rejected.")
            else: print("[INFO] No response.")

def main():
    node_id = NODE_ID if AGENT_ENABLED and NODE_ID else ""
    while not node_id:
        u = input(f"Enter ID (S1-S60 or {', '.join(PRIMARY_SENDERS)}): ").strip().upper()
        if re.match(r"^S([1-9]|[1-5][0-9]|60)$", u) or u in PRIMARY_SENDERS: node_id = u
    
    cleanup_old_files(node_id)
    if node_id in PRIMARY_SENDERS: BS_HOST, BS_PORT, pos, typ = BS1_HOST, BS1_PORT, (0,0), "primary_sender"
    else: BS_HOST, BS_PORT, pos, typ = BS1_HOST, BS1_PORT, (100,0), "sender"

    try:
        sock = socket.create_connection((BS_HOST, BS_PORT), timeout=5)
        sock.sendall((json.dumps({"type": typ, "id": node_id, "pos": list(pos)}) + "\n").encode("utf-8"))
        print(f"Registered {node_id}")
    except: return

    threading.Thread(target=receive_handler, args=(sock,), daemon=True).start()
    try: send_handler(sock, node_id)
    except: pass
    finally: stop_event.set(); export_all_results(node_id); sock.close()

if __name__ == "__main__":
    main()