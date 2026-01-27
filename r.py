# Dynamic_Receiver.py (r.py) - JAMMING VISUALIZATION EDITION
# - Visualizes Jammed/Corrupted packets in PDF Report
# - Adds "Noise" to constellation plots for jammed signals
# - Preserves all previous functionality

print("\n[DEBUG] RECEIVER: JAMMING VISUALIZATION ACTIVE.\n")

import socket, struct, threading, json, hashlib, random, re, os, time, datetime
from reedsolo import RSCodec
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
import numpy as np

# --- PLOTTING SETUP ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob

# ---------- Config ----------
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
log_lock = threading.Lock() 

# --- Session State ---
chat_partner = None
pending_request_from = None
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

def determine_M(msg): return 256 if len(msg) > 500 else (64 if len(msg) > 100 else 16)

def recv_full(sock, length):
    data = b""
    while len(data) < length:
        try:
            more = sock.recv(length - len(data))
            if not more: raise ConnectionError("socket closed")
            data += more
        except socket.timeout: return None
    return data

# ---------- PHY / Plotting ----------
def bits_from_bytes(b): return np.unpackbits(np.frombuffer(b, dtype=np.uint8))
def qam_mod(bits, M):
    k = int(np.log2(M))
    if len(bits) % k != 0:
        bits = np.concatenate([bits, np.zeros(k - (len(bits) % k), dtype=np.uint8)])
    ints = bits.reshape((-1, k)).dot(1 << np.arange(k - 1, -1, -1))
    sqrtM = int(np.sqrt(M))
    scale = np.sqrt((2.0 / 3.0) * (M - 1)) if M > 1 else 1.0
    return ((2*(ints%sqrtM)-(sqrtM-1))/scale) + 1j*((2*(ints//sqrtM)-(sqrtM-1))/scale)

def ofdm_mod(symbols, num_subcarriers, cp_len):
    if len(symbols) == 0: return np.array([])
    n_ofdm = int(np.ceil(len(symbols) / num_subcarriers))
    padded = np.pad(symbols, (0, n_ofdm * num_subcarriers - len(symbols)))
    ifft_data = np.fft.ifft(padded.reshape((n_ofdm, num_subcarriers)), axis=1)
    return np.hstack([ifft_data[:, -cp_len:], ifft_data]).flatten()

def make_constellation_plot(symbols, title, message_text, M, nc, cp):
    fig = plt.figure(figsize=(8,6))
    if len(symbols) > 0:
        # Visual styling: Jammed signals look red and messy
        color = 'red' if "[JAMMED]" in message_text else 'blue'
        alpha = 0.3 if "[JAMMED]" in message_text else 1.0
        plt.scatter(np.real(symbols), np.imag(symbols), s=10, c=color, alpha=alpha)
    plt.title(title); plt.grid(True)
    textstr = f'Msg: "{message_text[:30]}..."\nMod: QAM-{M}\nNC: {nc}, CP: {cp}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=props)
    return fig

def make_ofdm_plot(signal, title, message_text, M, nc, cp):
    fig = plt.figure(figsize=(8,4))
    if len(signal) > 0:
        limit = min(300, len(signal))
        plt.plot(np.real(signal[:limit]), label="I"); plt.plot(np.imag(signal[:limit]), label="Q")
    plt.title(title); plt.grid(True); plt.legend()
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

# ---------- Data Storage ----------
DATA_DIR = "node_logs"
os.makedirs(DATA_DIR, exist_ok=True)
messages_by_peer = {}
base_station_events = []

def cleanup_old_files(node_id):
    for f in glob.glob(os.path.join(DATA_DIR, f"{node_id}_*")):
        try: os.remove(f)
        except: pass

def log_event(txt): base_station_events.append(f"{datetime.datetime.utcnow().isoformat()}Z {txt}")

def record_message(peer_id, text_bytes, M, message_text, is_jammed=False):
    entry = {"ts": time.time(), "bytes": text_bytes, "M": M, "text": message_text}
    try:
        # Reconstruct PHY for visualization
        bits = bits_from_bytes(text_bytes)
        tx_symbols = qam_mod(bits, M)
        nc, cp = 64, 8
        
        if is_jammed:
            # --- JAMMING VISUALIZATION SIMULATION ---
            # Add severe Gaussian noise to valid symbols to simulate the visual effect of jamming
            noise = (np.random.normal(0, 0.5, tx_symbols.shape) + 
                     1j * np.random.normal(0, 0.5, tx_symbols.shape))
            tx_symbols = tx_symbols + noise
            # ----------------------------------------
            
        ofdm_sig = ofdm_mod(tx_symbols, nc, cp)
        entry["nc"], entry["cp"], entry["tx_symbols"], entry["ofdm_sig"] = nc, cp, tx_symbols, ofdm_sig
    except:
        entry["tx_symbols"], entry["ofdm_sig"] = np.array([]), np.array([])
        entry["nc"], entry["cp"] = 0, 0
    with log_lock:
        messages_by_peer.setdefault(peer_id, []).append(entry)

def export_all_results(node_id):
    print(f"\n[EXPORT] Generating plots...")
    figs = []
    with log_lock:
        peer_data = dict(messages_by_peer)
        
    for peer, entries in peer_data.items():
        for i, e in enumerate(entries):
            try:
                if e.get("tx_symbols").size > 0:
                    figs.append(make_constellation_plot(e["tx_symbols"], f"{peer} Constellation ({i})", e["text"], e["M"], e["nc"], e["cp"]))
                    figs.append(make_ofdm_plot(e["ofdm_sig"], f"{peer} OFDM ({i})", e["text"], e["M"], e["nc"], e["cp"]))
                    thresh = np.mean(np.abs(e["ofdm_sig"])**2) * 0.6 if e["ofdm_sig"].size > 0 else 0
                    figs.append(make_sensing_plot(e["ofdm_sig"], f"{peer} Sensing ({i})", e["text"], thresh))
            except Exception as e:
                print(f"[WARN] Failed to plot message from {peer}: {e}")
    
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pdf_path = os.path.join(DATA_DIR, f"{node_id}_plots_{timestamp}.pdf")
    save_plots_to_pdf(pdf_path, figs)
    
    with open(os.path.join(DATA_DIR, f"{node_id}_log_{timestamp}.txt"), "w") as f:
        for ev in base_station_events: f.write(ev + "\n")
        for peer, entries in peer_data.items():
            f.write(f"Peer: {peer}\n")
            for e in entries: f.write(f"Msg: {e['text']} (QAM-{e['M']})\n")
    print(f"[EXPORT] Saved plots to {pdf_path}")

def compute_ofdm_energy(message_bytes, M, nc, cp):
    bits = bits_from_bytes(message_bytes)
    tx_syms = qam_mod(bits, M)
    ofdm_sig = ofdm_mod(tx_syms, nc, cp)
    return (np.mean(np.abs(ofdm_sig)**2) if ofdm_sig.size>0 else 0.0), ofdm_sig

# ---------- Networking ----------
def send_message(sock, recipient, message_text):
    try:
        M = determine_M(message_text); nc, cp = 64, 8
        msg_bytes = message_text.encode("utf-8")
        energy, _ = compute_ofdm_energy(msg_bytes, M, nc, cp)

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
    global chat_partner, pending_request_from
    sock.settimeout(2)
    while not stop_event.is_set():
        try:
            hdr = recv_full(sock, 4)
            if not hdr: continue
            length = struct.unpack(">I", hdr)[0]
            data = recv_full(sock, length)
            
            src_len = struct.unpack(">H", data[:2])[0]
            src_id = data[2:2+src_len].decode("utf-8")
            pos = 2 + src_len + 8 
            
            try: 
                plaintext = aes_gcm_decrypt(rs.decode(data[pos:])[0], KEY)
                pkt = plaintext[5:]; msg_len = struct.unpack(">H", pkt[:2])[0]
                msg = pkt[2:2+msg_len].decode("utf-8")
            except: 
                # --- JAMMING HANDLING ---
                # If decryption fails, we assume it's a Jammed Packet
                print(f"\n[!] PACKET CORRUPTED/JAMMED FROM {src_id} [!]\n> ", end="", flush=True)
                # Record it as a "Jammed" entry (Random Noise Plot)
                # We make up dummy bytes since real ones are corrupted
                dummy_bytes = b'\x00' * 64 
                record_message(src_id, dummy_bytes, 16, "[JAMMED] CORRUPTED DATA", is_jammed=True)
                log_event(f"Received JAMMED/CORRUPTED packet from {src_id}")
                continue

            if msg == CTL_CONNECT_REQUEST:
                is_primary = src_id in PRIMARY_SENDERS
                with state_lock:
                    if chat_partner and is_primary and (chat_partner not in PRIMARY_SENDERS):
                        print(f"\n\n[PRIORITY] Primary '{src_id}' preempting Secondary '{chat_partner}'!")
                        send_message(sock, chat_partner, CTL_DISCONNECT)
                        chat_partner = None

                    if not chat_partner and not pending_request_from:
                        pending_request_from = src_id
                        print(f"\n[!] Request from {src_id}. 'accept'/'reject'?\n> ", end="", flush=True)
                continue
            
            if msg == CTL_DISCONNECT:
                 with state_lock:
                    if src_id == chat_partner: print(f"\n[INFO] {chat_partner} disconnected.\n> ", end="", flush=True); chat_partner = None
                 continue

            if msg == CTL_CONNECT_ACCEPT: pass 

            with state_lock:
                if src_id == chat_partner:
                    print(f"\n<<< {src_id}: '{msg}'\n> ", end="", flush=True)
                    record_message(src_id, msg.encode("utf-8"), determine_M(msg), msg)
                    log_event(f"Received from {src_id}: {msg}")

        except socket.timeout: continue
        except: continue

def main_loop(sock):
    global chat_partner, pending_request_from
    while not stop_event.is_set():
        try:
            with state_lock: curr_p = chat_partner; curr_req = pending_request_from
            prompt = "> " if curr_p else (f"Accept {curr_req}? (accept/reject): " if curr_req else "Waiting... ('exit'): ")
            user_input = input(prompt).strip()
            if not user_input: continue
            if user_input.lower() == 'exit':
                if curr_p: send_message(sock, curr_p, CTL_DISCONNECT)
                stop_event.set(); break

            if curr_req:
                if user_input.lower() == 'accept':
                    send_message(sock, curr_req, CTL_CONNECT_ACCEPT)
                    with state_lock: chat_partner = curr_req; pending_request_from = None
                    print(f"[SUCCESS] Connected with {chat_partner}.")
                else:
                    send_message(sock, curr_req, CTL_CONNECT_REJECT)
                    with state_lock: pending_request_from = None
                    print("[INFO] Rejected.")
            elif curr_p: send_message(sock, curr_p, user_input)

        except (EOFError, KeyboardInterrupt): stop_event.set()
        except Exception as e: print(f"[ERROR] {e}"); stop_event.set()

def main():
    node_id = ""
    while not re.match(r"^R([1-9]|[1-5][0-9]|60)$", node_id):
        node_id = input("Enter Receiver ID (R1-R60): ").strip().upper()
    
    cleanup_old_files(node_id)
    if node_id in PRIMARY_SENDERS: BS_HOST, BS_PORT = BS1_HOST, BS1_PORT
    else: BS_HOST, BS_PORT = BS1_HOST, BS1_PORT

    try:
        sock = socket.create_connection((BS_HOST, BS_PORT), timeout=5)
        sock.sendall((json.dumps({"type": "receiver", "id": node_id, "pos": (random.uniform(0,100), 0)}) + "\n").encode("utf-8"))
        print(f"Registered {node_id}")
    except: return

    threading.Thread(target=receive_handler, args=(sock,), daemon=True).start()
    try: main_loop(sock)
    finally: stop_event.set(); export_all_results(node_id); sock.close()

if __name__ == "__main__":
    main()