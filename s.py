# Dynamic_Sender.py (Enhanced + Agent) - FIXED with REAL spectrum sensing
# - Proper energy detection spectrum sensing with noise calibration
# - Statistical detection theory with false alarm probability
# - Actual channel measurement, not self-deferral nonsense
import random

import socket, struct, threading, json, hashlib, random, re, os, time, datetime
from reedsolo import RSCodec
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import scipy.special

# ---------- CONFIG: Base Stations ----------
BS1_HOST, BS1_PORT = "127.0.0.1", 50050
BS2_HOST, BS2_PORT = "127.0.0.1", 50051

# ---------- Primary / Secondary setup ----------
PRIMARY_SENDERS = ["JAZZ", "UFONE", "TELENOR", "WARID", "STARLINK", "ZONG", "SCO", "PTCL"]

# ---------- AGENT / NODE CONFIG ----------
NODE_ID = "S5"                      
NODE_POS = (220.0, 0.0)             
DEFAULT_RECIPIENT = "R1"            
AGENT_ENABLED = False               
AGENT_INTERVAL = 15.0               
AGENT_PAYLOADS = ["Hello from agent", "Telemetry sample", "ping"] 

# ---------- SPECTRUM SENSING CONFIG ----------
SENSING_WINDOW_SIZE = 1024  # Samples for sensing
NOISE_FLOOR = -90  # dBm (typical receiver noise floor)
SENSING_THRESHOLD_DB = -85  # dBm (5dB above noise floor)
FALSE_ALARM_PROB = 0.1  # P_fa = 10%
SENSING_TIME = 0.1  # seconds for sensing duration

# ---------- Crypto / FEC ----------
PASSPHRASE = b"very secret passphrase - change this!"
KEY = hashlib.sha256(PASSPHRASE).digest()
rs = RSCodec(40)
stop_event = threading.Event()
plot_lock = threading.Lock()

# --- Session Management ---
chat_partner = None
connection_status = threading.Event()
connection_accepted = False
state_lock = threading.Lock()

# Control messages
CTL_CONNECT_REQUEST = "__CONNECT_REQUEST__"
CTL_CONNECT_ACCEPT = "__CONNECT_ACCEPT__"
CTL_CONNECT_REJECT = "__CONNECT_REJECT__"
CTL_DISCONNECT = "__DISCONNECT__"
CTL_SENSE_QUERY = "__SENSE_QUERY__"
CTL_SENSE_REPLY = "__SENSE_REPLY__"

# ---------- REAL SPECTRUM SENSING FUNCTIONS ----------
def generate_awgn_noise(num_samples, snr_db=20):
    """Generate AWGN noise for channel simulation"""
    signal_power = 1.0  # Normalized signal power
    snr_linear = 10**(snr_db/10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(num_samples) + 1j*np.random.randn(num_samples))
    return noise

def simulate_channel_occupancy():
    """Simulate actual channel occupancy (primary user activity)"""
    # In real system, this would be actual RF sampling
    # Here we simulate with 30% probability of primary user present
    primary_present = random.random() < 0.3
    
    if primary_present:
        # Generate primary user signal (QPSK-like)
        primary_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], SENSING_WINDOW_SIZE)
        primary_signal = primary_symbols * 0.8  # Scale for realistic power
        noise = generate_awgn_noise(SENSING_WINDOW_SIZE, snr_db=15)
        received_signal = primary_signal + noise
    else:
        # Noise only
        received_signal = generate_awgn_noise(SENSING_WINDOW_SIZE, snr_db=20)
    
    return received_signal, primary_present

def calculate_energy_detection_threshold(samples, p_fa=FALSE_ALARM_PROB):
    """Calculate proper detection threshold using statistical theory"""
    N = len(samples)
    
    # Estimate noise variance from samples (assuming mostly noise)
    noise_variance = np.var(samples)
    
    # For complex signals, variance is doubled
    noise_variance_complex = noise_variance * 2 if np.iscomplexobj(samples) else noise_variance
    
    # Energy detection threshold formula: 
    # threshold = noise_variance * (1 + Q^{-1}(P_fa) / sqrt(N/2))
    Q_inv = np.sqrt(2) * scipy.special.erfinv(1 - 2 * p_fa)
    threshold = noise_variance_complex * (1 + Q_inv / np.sqrt(N/2))
    
    return threshold, noise_variance_complex

def perform_spectrum_sensing():
    """Perform actual energy detection spectrum sensing"""
    # Simulate receiving actual channel samples
    channel_samples, actual_primary_present = simulate_channel_occupancy()
    
    # Calculate test statistic (energy)
    test_statistic = np.sum(np.abs(channel_samples)**2) / len(channel_samples)
    
    # Calculate proper detection threshold
    threshold, noise_variance = calculate_energy_detection_threshold(channel_samples)
    
    # Decision
    channel_busy = test_statistic > threshold
    
    sensing_result = {
        'busy': channel_busy,
        'test_statistic': test_statistic,
        'threshold': threshold,
        'noise_variance': noise_variance,
        'snr_estimate': 10 * np.log10(test_statistic / noise_variance) if noise_variance > 0 else -100,
        'actual_primary': actual_primary_present,
        'samples': channel_samples
    }
    
    return sensing_result

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

def make_sensing_plot(sensing_result, title, message_text):

    """Plot actual spectrum sensing results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot 1: Time domain samples
    samples = sensing_result.get('samples', np.array([]))
    # s.py

# ...
    if samples.size > 0:
        

        t = np.arange(len(samples))
        # Slice 't' to match the 'samples' slice
        ax1.plot(t[:200], np.real(samples[:200]), label='I', alpha=0.7) # <--- FIXED
        ax1.plot(t[:200], np.imag(samples[:200]), label='Q', alpha=0.7) # <--- FIXED
        ax1.set_title(f"Channel Samples - {title}")
# ...
    # Plot 2: Energy detection results
    test_stat = sensing_result.get('test_statistic', 0)
    threshold = sensing_result.get('threshold', 0)
    busy = sensing_result.get('busy', False)
    actual_primary = sensing_result.get('actual_primary', False)
    
    ax2.bar(['Noise Floor', 'Test Statistic', 'Threshold'], 
            [sensing_result.get('noise_variance', 0), test_stat, threshold],
            color=['blue', 'red' if busy else 'green', 'orange'])
    ax2.set_ylabel('Power')
    ax2.set_title(f"Sensing Result: {'BUSY' if busy else 'IDLE'} | "
                  f"Actual: {'PRIMARY' if actual_primary else 'CLEAR'} | "
                  f"SNR: {sensing_result.get('snr_estimate', 0):.1f} dB")
    ax2.grid(True)
    
    plt.tight_layout()
    
    textstr = f'Message: "{message_text[:50]}{"..." if len(message_text)>50 else ""}"\n' \
              f'Detection: {busy}\nP_FA: {FALSE_ALARM_PROB}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.figtext(0.02, 0.02, textstr, fontsize=9, bbox=props)
    
    return fig

def save_plots_to_pdf(pdf_path, figs):
    with plot_lock:
        with PdfPages(pdf_path) as pdf:
            for f in figs:
                pdf.savefig(f)
                plt.close(f)

# ---------- Agent-local decision helpers ----------
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
sensing_results = []  # Store actual sensing results

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

def record_message(recipient, text_bytes, M, message_text, sensing_result=None):
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
        "ts": time.time(),
        "sensing_result": sensing_result
    })
    if sensing_result:
        sensing_results.append(sensing_result)
    print(f"[PHY_PARAMS] M={M}, Subcarriers={nc}, CP={cp} for message: '{message_text[:30]}...'")

# ---------- SNR vs BER sim ----------
# s.py

# REPLACE your old compute_ber_for_snr with this one:

def compute_ber_for_snr(tx_bits, M, snr_db):
    """
    Computes BER by simulating a full QAM Mod -> AWGN -> QAM Demod chain.
    """
    if tx_bits is None or tx_bits.size == 0:
        return 1.0
    
    # 1. Modulate
    symbols = qam_mod(tx_bits, M)
    if symbols.size == 0:
        return 1.0

    # 2. Add Noise
    snr_linear = 10**(snr_db / 10.0)
    # Calculate noise variance based on signal power (which is normalized to ~1.0)
    signal_power = np.mean(np.abs(symbols)**2)
    noise_var = signal_power / snr_linear
    
    # Generate complex noise
    noise = (np.sqrt(noise_var / 2.0) * (np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape)))
    
    # 3. Receive Noisy Signal
    rx = symbols + noise
    
    # 4. Demodulate
    # We must slice tx_bits to match the number of bits that will be demodulated
    # as qam_mod may pad the input.
    num_bits_to_compare = int(symbols.size * np.log2(M))
    if num_bits_to_compare > tx_bits.size:
         # This happens if padding occurred. We compare against the padded bits.
         k = int(np.log2(M))
         pad_len = k - (tx_bits.size % k) if (tx_bits.size % k) != 0 else 0
         tx_bits_padded = np.concatenate([tx_bits, np.zeros(pad_len, dtype=np.uint8)])
         bits_to_compare = tx_bits_padded[:num_bits_to_compare]
    else:
        bits_to_compare = tx_bits[:num_bits_to_compare]

    rx_bits = qam_demod(rx, M)
    
    # Ensure arrays are the same length for comparison
    compare_len = min(len(bits_to_compare), len(rx_bits))
    if compare_len == 0:
        return 1.0
        
    bits_to_compare = bits_to_compare[:compare_len]
    rx_bits = rx_bits[:compare_len]

    # 5. Calculate Bit Errors
    num_errors = np.sum(bits_to_compare != rx_bits)
    ber = num_errors / compare_len
    
    return ber

# s.py

# ... (other functions) ...





# s.py

# ... (other functions) ...

def export_all_results(node_id):
    figs = []
    for i, msg in enumerate(messages_sent):
        tx_syms, ofdm_sig = msg["tx_syms"], msg["ofdm"]
        message_text = msg.get("text", "Unknown message")
        M, nc, cp = msg["M"], msg["nc"], msg["cp"]
        
        figs.append(make_constellation_plot(
            tx_syms,
            f"{msg['recipient']} - Constellation Diagram ({i})",
            message_text, M, nc, cp
        ))
        figs.append(make_ofdm_plot(
            ofdm_sig,
            f"{msg['recipient']} - OFDM I/Q Signal ({i})",
            message_text, M, nc, cp
        ))
        
        # REAL spectrum sensing plot
        sensing_result = msg.get("sensing_result")
        if sensing_result:
            figs.append(make_sensing_plot(
                sensing_result,
                f"{msg['recipient']} - Spectrum Sensing ({i})",
                message_text
            ))
    
    if messages_sent:
        # --- START: THIS IS THE UPDATED SECTION ---
        # Get modulation from the first message
        M = messages_sent[0]["M"]
        
        # Combine ALL sent messages into one giant bitstream
        print(f"[EXPORT] Combining {len(messages_sent)} sent messages for BER test...")
        
        all_message_bytes = b"".join([msg["bytes"] for msg in messages_sent])
        bits = bits_from_bytes(all_message_bytes)
        
        num_bits_tested = bits.size
        print(f"[EXPORT] Total 'actual' bits for testing: {num_bits_tested}")
        
        message_text = f"Actual data ({len(messages_sent)} msgs, {num_bits_tested} bits)"
        snrs = np.arange(0, 21, 2)
        
        if num_bits_tested < 1000:
            print("[EXPORT] WARNING: Still not enough bits for a smooth curve. Send more messages!")
        
        print("[EXPORT] Running BER simulation on 'actual' data...")
        bers = [compute_ber_for_snr(bits, M, s) for s in snrs]
        print("[EXPORT] ...simulation complete.")
        # --- END: UPDATED SECTION ---

        snr_fig = plt.figure(figsize=(8, 6))
        plt.semilogy(snrs, bers, marker='o', linewidth=2, markersize=8)
        plt.title("SNR vs BER Performance")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate (BER)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        ber_at_10db_index = np.where(snrs == 10)[0][0]
        ber_at_20db_index = np.where(snrs == 20)[0][0]
        textstr = (
            f'Message: "{message_text[:50]}{"..." if len(message_text) > 50 else ""}"\n'
            f"Modulation: QAM-{M}\n"
            f"BER@10dB: {bers[ber_at_10db_index]:.2e}\n"
            f"BER@20dB: {bers[ber_at_20db_index]:.2e}"
        )
        
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
        f.write("=" * 50 + "\n\n")
        
        for ev in base_station_events:
            f.write(ev + "\n")
        
        f.write("\n--- Messages Sent ---\n")
        for m in messages_sent:
            f.write(f"Time: {datetime.datetime.utcfromtimestamp(m['ts']).isoformat()}\n")
            f.write(f"To: {m['recipient']} | Length: {len(m['bytes'])} bytes\n")
            f.write(f"Modulation: QAM-{m['M']} | Subcarriers: {m['nc']} | CP: {m['cp']}\n")
            f.write(f"Sensing: {m.get('sensing_result', 'N/A')}\n")  # Fixed line
            f.write(f"Message: {m.get('text', 'Unknown')}\n")
            f.write("-" * 40 + "\n")
    
    print(f"[EXPORT] Saved plots to {pdf_path} and logs to {log_path}")
    log_event(f"Exported plots to {pdf_path} and logs to {log_path}")

# ---------- REAL Spectrum Sensing ----------
def compute_ofdm_energy_from_message_bytes(message_bytes, M, nc, cp):
    """Build OFDM from message bytes and compute average energy (power)."""
    bits = bits_from_bytes(message_bytes)
    tx_syms = qam_mod(bits, M)
    ofdm_sig = ofdm_mod(tx_syms, nc, cp)
    energy = np.mean(np.abs(ofdm_sig)**2) if ofdm_sig.size>0 else 0.0
    return energy, ofdm_sig

# ---------- Networking handlers ----------
def send_message(sock, recipient, message_text):
    """Helper to construct and send a message with REAL spectrum sensing."""
    global chat_partner
    try:
        M = determine_M(message_text)
        nc, cp = pick_num_subcarriers_local(message_text, M)

        # Determine primary/secondary by NODE_ID presence in PRIMARY_SENDERS
        is_primary = NODE_ID.upper() in PRIMARY_SENDERS

        # Control messages bypass sensing (always send)
        control_msgs = [CTL_CONNECT_REQUEST, CTL_CONNECT_ACCEPT, CTL_CONNECT_REJECT, CTL_DISCONNECT]
        bypass_sensing = (message_text in control_msgs) or is_primary

        # REAL SPECTRUM SENSING for secondary users
        sensing_result = None
        if not bypass_sensing:
            print(f"[SENSING] Secondary node {NODE_ID} performing spectrum sensing...")
            sensing_result = perform_spectrum_sensing()
            
            if sensing_result['busy']:
                print(f"[SENSING] Channel BUSY - deferring transmission (SNR: {sensing_result['snr_estimate']:.1f} dB)")
                log_event(f"Deferred send to {recipient}: Channel BUSY (SNR: {sensing_result['snr_estimate']:.1f} dB)")
                return False, sensing_result  # indicate not sent with sensing result
            else:
                print(f"[SENSING] Channel IDLE - clear to transmit (SNR: {sensing_result['snr_estimate']:.1f} dB)")
                log_event(f"Channel IDLE - transmitting to {recipient}")

        # Compute energy for logging (not for sensing decision)
        msg_bytes = message_text.encode("utf-8")
        energy, ofdm_sig = compute_ofdm_energy_from_message_bytes(msg_bytes, M, nc, cp)

        # Compose and send message
        pkt = struct.pack(">H", len(message_text.encode())) + message_text.encode()
        plaintext = struct.pack(">H H B", M, nc, cp) + pkt
        enc = aes_gcm_encrypt(plaintext, KEY)
        enc_rs = rs.encode(enc)

        # Include sensing metadata if available
        dst_b = recipient.encode("utf-8")
        sensing_energy = sensing_result['test_statistic'] if sensing_result else energy
        sensing_header = struct.pack(">d", float(sensing_energy))
        wire_payload = struct.pack(">H", len(dst_b)) + dst_b + sensing_header + enc_rs

        sock.sendall(struct.pack(">I", len(wire_payload)) + wire_payload)

        # Log and record the message with sensing results
        if message_text not in control_msgs:
             record_message(recipient, msg_bytes, M, message_text, sensing_result)
        log_event(f"Sent to {recipient}: {message_text} (M={M}, NC={nc}, CP={cp}, primary={is_primary})")
        return True, sensing_result
    except Exception as e:
        print(f"[ERROR] Failed to send message: {e}")
        log_event(f"Send error: {e}")
        with state_lock:
            chat_partner = None
        return False, None


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
                print(f"[AGENT] Performing spectrum sensing before transmission...")
                success, sensing_result = send_message(sock, DEFAULT_RECIPIENT, payload)
                if success:
                    print(f"[AGENT] Sent to {DEFAULT_RECIPIENT}: '{payload}'")
                else:
                    print(f"[AGENT] Transmission deferred - channel busy")
            
            # Wait for the interval, checking for stop event periodically
            for _ in range(int(AGENT_INTERVAL * 2)):
                if stop_event.is_set(): break
                time.sleep(0.5)
                
        except Exception as e:
            if not stop_event.is_set():
                log_event(f"Agent error: {e}")
                print(f"[AGENT] Error: {e}")
            break
# s.py

# ... (put this near your other PHY helpers like qam_mod)

def qam_demod(rx_symbols, M):
    """Demodulates noisy QAM symbols (hard-decision)"""
    if M <= 1:
        return np.array([], dtype=np.uint8)
        
    k = int(np.log2(M))
    sqrtM = int(np.sqrt(M))
    
    # 1. Create the ideal constellation grid
    # (2*i - (sqrtM-1))
    x_int_ideal = np.arange(sqrtM)
    y_int_ideal = np.arange(sqrtM)
    raw_x_ideal = 2 * x_int_ideal - (sqrtM - 1)
    raw_y_ideal = 2 * y_int_ideal - (sqrtM - 1)
    
    scale = np.sqrt((2.0 / 3.0) * (M - 1)) if M > 1 else 1.0
    
    constellation_x = raw_x_ideal / scale
    constellation_y = raw_y_ideal / scale
    
    # 2. De-normalize the received symbols
    rx_x_scaled = np.real(rx_symbols) * scale
    rx_y_scaled = np.imag(rx_symbols) * scale

    # 3. Find the closest ideal I/Q point for each received symbol
    # (Hard-decision decoding)
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    rx_x_int = np.array([find_nearest(raw_x_ideal, x) for x in rx_x_scaled])
    rx_y_int = np.array([find_nearest(raw_y_ideal, y) for y in rx_y_scaled])

    # 4. Convert (x_int, y_int) back to integers
    rx_ints = rx_y_int * sqrtM + rx_x_int

    # 5. Convert integers back to bits
    rx_bits_flat = np.unpackbits(rx_ints.astype(np.uint8).reshape(-1, 1), axis=1)
    
    # We only care about the 'k' bits for each symbol
    # Bits are unpacked in MSB-first order (e.g., [0,0,0,0,1,1,0,1] for 13)
    # We need the LSBs
    rx_bits = rx_bits_flat[:, -k:]
    
    return rx_bits.flatten().astype(np.uint8)

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
            
            success, sensing_result = send_message(sock, chat_partner, message)
            if not success:
                print("[SENSING] Transmission deferred - channel busy")
        
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