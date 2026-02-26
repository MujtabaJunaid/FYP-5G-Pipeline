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
from mpl_toolkits.mplot3d import Axes3D
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

def make_psd_plot(ofdm_sig, title, message_text):
    """Power Spectral Density via FFT"""
    fig = plt.figure(figsize=(8,4))
    if ofdm_sig.size > 0:
        fft_result = np.abs(np.fft.fft(ofdm_sig))**2
        freq = np.fft.fftfreq(len(fft_result))
        plt.semilogy(freq[:len(freq)//2], fft_result[:len(fft_result)//2], color='darkblue')
    plt.title(title); plt.xlabel("Normalized Frequency"); plt.ylabel("PSD (dB)")
    plt.grid(True, alpha=0.3)
    return fig

def make_received_energy_histogram(messages_by_peer_dict):
    """Energy distribution of received messages per peer"""
    fig = plt.figure(figsize=(10,5))
    peer_energies = {}
    for peer, entries in messages_by_peer_dict.items():
        for e in entries:
            if e.get("ofdm_sig").size > 0:
                energy = float(np.mean(np.abs(e["ofdm_sig"])**2))
                if peer not in peer_energies: peer_energies[peer] = []
                peer_energies[peer].append(energy)
    
    if peer_energies:
        peers = list(peer_energies.keys())
        avg_energies = [np.mean(peer_energies[p]) for p in peers]
        colors = ['red' if p in PRIMARY_SENDERS else 'blue' for p in peers]
        plt.bar(peers, avg_energies, color=colors, alpha=0.7)
    plt.title("Average Received Energy by Peer"); plt.ylabel("Energy (W)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    return fig

def make_jammed_vs_clean_comparison(messages_by_peer_dict):
    """Compare jammed vs clean packets"""
    fig = plt.figure(figsize=(10,6))
    jammed_count = sum(1 for peer, entries in messages_by_peer_dict.items() 
                      for e in entries if "[JAMMED]" in e.get("text", ""))
    clean_count = sum(1 for peer, entries in messages_by_peer_dict.items() 
                     for e in entries if "[JAMMED]" not in e.get("text", ""))
    
    categories = ["Clean Packets", "Jammed Packets"]
    counts = [clean_count, jammed_count]
    colors = ['green', 'red']
    plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    plt.title("Packet Reception Status: Clean vs Jammed"); plt.ylabel("Count")
    for i, v in enumerate(counts):
        plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    return fig

def make_packets_per_peer(messages_by_peer_dict):
    """Message count distribution per peer"""
    fig = plt.figure(figsize=(10,5))
    if messages_by_peer_dict:
        peers = list(messages_by_peer_dict.keys())
        counts = [len(messages_by_peer_dict[p]) for p in peers]
        colors = ['red' if p in PRIMARY_SENDERS else 'blue' for p in peers]
        plt.bar(peers, counts, color=colors, alpha=0.7)
    plt.title("Message Count per Peer"); plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    return fig

def make_symbol_scatter_3d(symbols, title, M):
    """3D constellation with magnitude and phase info"""
    fig = plt.figure(figsize=(10,7))
    if len(symbols) > 0:
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.real(symbols), np.imag(symbols)
        z = np.abs(symbols)
        scatter = ax.scatter(x, y, z, c=np.angle(symbols), cmap='hsv', s=20)
        ax.set_xlabel('I'); ax.set_ylabel('Q'); ax.set_zlabel('Magnitude')
        ax.set_title(title)
        plt.colorbar(scatter, label='Phase (rad)')
    return fig

def make_amplitude_histogram(symbols, title):
    """Amplitude distribution of received symbols"""
    fig = plt.figure(figsize=(8,5))
    if len(symbols) > 0:
        amplitudes = np.abs(symbols)
        plt.hist(amplitudes, bins=20, color='purple', alpha=0.7, edgecolor='black')
    plt.title(title); plt.xlabel("Amplitude"); plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3, axis='y')
    return fig

def make_phase_distribution(symbols, title):
    """Phase angle distribution of received symbols"""
    fig = plt.figure(figsize=(8,5))
    if len(symbols) > 0:
        phases = np.angle(symbols)
        plt.hist(phases, bins=16, color='orange', alpha=0.7, edgecolor='black')
    plt.title(title); plt.xlabel("Phase (rad)"); plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3, axis='y')
    return fig

def make_snr_estimate_plot(ofdm_sig, M, title):
    """Estimate SNR from received signal"""
    fig = plt.figure(figsize=(8,5))
    if ofdm_sig.size > 0:
        power = np.mean(np.abs(ofdm_sig)**2)
        noise_est = power * 0.01
        snr_db = 10*np.log10(power/max(noise_est, 1e-12))
        plt.text(0.5, 0.5, f'Est. SNR: {snr_db:.2f} dB\nModulation: QAM-{M}\nReceived Power: {power:.2e}',
                transform=plt.gca().transAxes, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    plt.title(title); plt.axis('off')
    return fig

def make_packet_timeline(messages_by_peer_dict):
    """Timeline of received packets with jammer events"""
    fig = plt.figure(figsize=(12,6))
    all_packets = []
    for peer, entries in messages_by_peer_dict.items():
        for e in entries:
            is_jammed = "[JAMMED]" in e.get("text", "")
            all_packets.append((e.get("ts", 0), peer, is_jammed))
    
    if all_packets:
        all_packets.sort(key=lambda x: x[0])
        min_ts = all_packets[0][0]
        times = [(p[0] - min_ts) for p in all_packets]
        colors = ['red' if p[2] else 'blue' for p in all_packets]
        labels = [p[1] for p in all_packets]
        plt.scatter(times, range(len(all_packets)), c=colors, s=100, alpha=0.6)
    
    plt.title("Received Packet Timeline (Blue=Clean, Red=Jammed)"); plt.xlabel("Time (s)")
    plt.ylabel("Packet Index"); plt.grid(True, alpha=0.3, axis='x')
    return fig

def make_corruption_pattern(messages_by_peer_dict):
    """Show pattern of jammed vs clean packets over time"""
    fig = plt.figure(figsize=(12,4))
    all_packets = []
    for peer, entries in messages_by_peer_dict.items():
        for e in entries:
            is_jammed = "[JAMMED]" in e.get("text", "")
            all_packets.append((e.get("ts", 0), 1 if is_jammed else 0))
    
    if all_packets:
        all_packets.sort(key=lambda x: x[0])
        min_ts = all_packets[0][0]
        times = [(p[0] - min_ts) for p in all_packets]
        corruption = [p[1] for p in all_packets]
        plt.bar(range(len(corruption)), corruption, color=['red' if c else 'green' for c in corruption], alpha=0.7)
    
    plt.title("Packet Corruption Timeline"); plt.xlabel("Packet #"); plt.ylabel("Jammed (1) / Clean (0)")
    plt.yticks([0, 1], ['Clean', 'Jammed'])
    plt.grid(True, alpha=0.3, axis='x')
    return fig

def make_spectrum_before_after_jamming(messages_by_peer_dict):
    """Compare spectrum before and after jamming events"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
    
    clean_packets = []
    jammed_packets = []
    
    for peer, entries in messages_by_peer_dict.items():
        for e in entries:
            if "[JAMMED]" in e.get("text", ""):
                jammed_packets.append(e)
            else:
                clean_packets.append(e)
    
    # Before jamming (clean packets)
    if clean_packets:
        for pkt in clean_packets[:3]:  # Average first 3 clean packets
            if pkt.get("ofdm_sig", np.array([])).size > 0:
                fft_vec = np.abs(np.fft.fft(pkt["ofdm_sig"][:256]))**2
                ax1.semilogy(fft_vec[:128], alpha=0.6, linewidth=1)
        ax1.set_title("BEFORE JAMMING: Clean Packet Spectra"); ax1.set_ylabel("PSD (dB)")
        ax1.grid(True, alpha=0.3)
    
    # After jamming (jammed packets)
    if jammed_packets:
        for pkt in jammed_packets[:3]:  # Average first 3 jammed packets
            if pkt.get("ofdm_sig", np.array([])).size > 0:
                fft_vec = np.abs(np.fft.fft(pkt["ofdm_sig"][:256]))**2
                ax2.semilogy(fft_vec[:128], alpha=0.6, linewidth=1, color='red')
        ax2.set_title("AFTER JAMMING: Corrupted Packet Spectra"); ax2.set_ylabel("PSD (dB)")
        ax2.set_xlabel("Frequency Bin")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def make_spectrum_occupancy_comparison(messages_by_peer_dict):
    """Spectrum occupancy before vs after jamming"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    
    clean_psd = None
    jammed_psd = None
    
    # Calculate average clean spectrum
    clean_count = 0
    for peer, entries in messages_by_peer_dict.items():
        for e in entries:
            if "[JAMMED]" not in e.get("text", "") and e.get("ofdm_sig", np.array([])).size > 0:
                fft_vec = np.abs(np.fft.fft(e["ofdm_sig"][:256]))**2
                if clean_psd is None:
                    clean_psd = fft_vec.copy()
                else:
                    clean_psd += fft_vec
                clean_count += 1
    
    if clean_count > 0:
        clean_psd /= clean_count
        ax1.fill_between(range(128), clean_psd[:128], alpha=0.6, color='green', label='Signal')
        ax1.plot(range(128), clean_psd[:128], 'g-', linewidth=2)
        ax1.set_title("Clean Channel Spectrum Occupancy"); ax1.set_ylabel("Power")
        ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # Calculate average jammed spectrum
    jammed_count = 0
    for peer, entries in messages_by_peer_dict.items():
        for e in entries:
            if "[JAMMED]" in e.get("text", "") and e.get("ofdm_sig", np.array([])).size > 0:
                fft_vec = np.abs(np.fft.fft(e["ofdm_sig"][:256]))**2
                if jammed_psd is None:
                    jammed_psd = fft_vec.copy()
                else:
                    jammed_psd += fft_vec
                jammed_count += 1
    
    if jammed_count > 0:
        jammed_psd /= jammed_count
        ax2.fill_between(range(128), jammed_psd[:128], alpha=0.6, color='red', label='Jamming+Signal')
        ax2.plot(range(128), jammed_psd[:128], 'r-', linewidth=2)
        ax2.set_title("Jammed Channel Spectrum Occupancy"); ax2.set_ylabel("Power")
        ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def make_jamming_intensity_plot(messages_by_peer_dict):
    """Jamming intensity over time showing attack footprint"""
    fig = plt.figure(figsize=(12,5))
    
    timestamps = []
    intensities = []
    packet_types = []
    
    for peer, entries in messages_by_peer_dict.items():
        for e in entries:
            ts = e.get("ts", 0)
            is_jammed = "[JAMMED]" in e.get("text", "")
            
            if e.get("ofdm_sig", np.array([])).size > 0:
                # Calculate "jamming intensity" as deviation from ideal signal
                fft_vec = np.abs(np.fft.fft(e["ofdm_sig"][:256]))**2
                intensity = np.std(fft_vec) / (np.mean(fft_vec) + 1e-12)
                
                timestamps.append(ts)
                intensities.append(intensity)
                packet_types.append('jammed' if is_jammed else 'clean')
    
    if timestamps:
        min_ts = min(timestamps)
        rel_times = [(t - min_ts) for t in timestamps]
        colors = ['red' if ptype == 'jammed' else 'blue' for ptype in packet_types]
        
        plt.scatter(rel_times, intensities, c=colors, s=80, alpha=0.6, edgecolor='black', linewidth=0.5)
        plt.axhline(y=np.mean(intensities), color='gray', linestyle='--', label='Mean Intensity')
    
    plt.title("Jamming Intensity Profile (Red=Jammed, Blue=Clean)")
    plt.xlabel("Time (s)"); plt.ylabel("Spectral Intensity")
    plt.legend(); plt.grid(True, alpha=0.3)
    return fig

def make_sinr_comparison_plot(messages_by_peer_dict):
    """SINR degradation showing jamming impact"""
    fig = plt.figure(figsize=(12,5))
    
    clean_sinrs = []
    jammed_sinrs = []
    
    for peer, entries in messages_by_peer_dict.items():
        for e in entries:
            if e.get("ofdm_sig", np.array([])).size > 0:
                signal_power = np.mean(np.abs(e["ofdm_sig"])**2)
                noise_est = signal_power * 0.01
                
                # For jammed packets, add interference component
                jammer_power = signal_power * 0.5 if "[JAMMED]" in e.get("text", "") else 0
                
                sinr_linear = signal_power / max(jammer_power + noise_est, 1e-12)
                sinr_db = 10 * np.log10(sinr_linear)
                
                if "[JAMMED]" in e.get("text", ""):
                    jammed_sinrs.append(sinr_db)
                else:
                    clean_sinrs.append(sinr_db)
    
    # Box plot comparison
    data_to_plot = [clean_sinrs, jammed_sinrs]
    labels = ['Clean Packets', 'Jammed Packets']
    colors_box = ['green', 'red']
    
    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.title("SINR Distribution: Clean vs Jammed Packets")
    plt.ylabel("SINR (dB)"); plt.grid(True, alpha=0.3, axis='y')
    return fig

def make_band_occupancy_evolution(messages_by_peer_dict):
    """Band occupancy changes over time during jamming"""
    fig = plt.figure(figsize=(12,6))
    
    occupancy_timeline = []
    timestamps = []
    
    for peer, entries in messages_by_peer_dict.items():
        for e in entries:
            if e.get("ofdm_sig", np.array([])).size > 0:
                fft_vec = np.abs(np.fft.fft(e["ofdm_sig"][:128]))**2
                
                # Calculate occupancy as % of subcarriers above threshold
                threshold = np.max(fft_vec) * 0.1
                occupancy_pct = 100 * np.sum(fft_vec > threshold) / len(fft_vec)
                
                occupancy_timeline.append(occupancy_pct)
                timestamps.append(e.get("ts", 0))
    
    if timestamps:
        min_ts = min(timestamps)
        rel_times = [(t - min_ts) for t in timestamps]
        
        # Color code by jamming state
        colors = ['red' if i > np.median(occupancy_timeline) * 1.2 else 'blue' for i in occupancy_timeline]
        
        plt.scatter(rel_times, occupancy_timeline, c=colors, s=100, alpha=0.6, edgecolor='black')
        plt.plot(rel_times, occupancy_timeline, 'k--', alpha=0.3, linewidth=1)
        plt.axhline(y=np.median(occupancy_timeline), color='orange', linestyle='--', label='Baseline')
    
    plt.title("Spectrum Band Occupancy Evolution (Red=High Occupancy/Jamming)")
    plt.xlabel("Time (s)"); plt.ylabel("Occupied Subcarriers (%)")
    plt.legend(); plt.grid(True, alpha=0.3)
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
    print(f"\n[EXPORT] Generating comprehensive receiver plots...")
    figs = []
    with log_lock:
        peer_data = dict(messages_by_peer)
    
    # 1. Individual message plots
    for peer, entries in peer_data.items():
        for i, e in enumerate(entries):
            try:
                if e.get("tx_symbols", np.array([])).size > 0:
                    status = "JAMMED" if "[JAMMED]" in e["text"] else "CLEAN"
                    figs.append(make_constellation_plot(e["tx_symbols"], f"[{peer}] [{status}] Constellation ({i})", e["text"], e["M"], e["nc"], e["cp"]))
                    figs.append(make_ofdm_plot(e["ofdm_sig"], f"[{peer}] [{status}] OFDM Time Domain ({i})", e["text"], e["M"], e["nc"], e["cp"]))
                    thresh = np.mean(np.abs(e["ofdm_sig"])**2) * 0.6 if e["ofdm_sig"].size > 0 else 0
                    figs.append(make_sensing_plot(e["ofdm_sig"], f"[{peer}] [{status}] Power Envelope ({i})", e["text"], thresh))
                    figs.append(make_psd_plot(e["ofdm_sig"], f"[{peer}] [{status}] PSD ({i})", e["text"]))
                    figs.append(make_snr_estimate_plot(e["ofdm_sig"], e["M"], f"[{peer}] [{status}] SNR ({i})"))
                    if len(e["tx_symbols"]) > 0:
                        figs.append(make_amplitude_histogram(e["tx_symbols"], f"[{peer}] Amplitude Dist ({i})"))
                        figs.append(make_phase_distribution(e["tx_symbols"], f"[{peer}] Phase Dist ({i})"))
            except Exception as ex:
                print(f"[WARN] Failed to plot message from {peer}: {ex}")
    
    # 2. Aggregate receiver analysis
    try:
        if peer_data:
            figs.append(make_received_energy_histogram(peer_data))
            figs.append(make_jammed_vs_clean_comparison(peer_data))
            figs.append(make_packets_per_peer(peer_data))
            figs.append(make_packet_timeline(peer_data))
            figs.append(make_corruption_pattern(peer_data))
            
            # 3. SPECTRUM SENSING & JAMMING ANALYSIS
            figs.append(make_spectrum_before_after_jamming(peer_data))
            figs.append(make_spectrum_occupancy_comparison(peer_data))
            figs.append(make_jamming_intensity_plot(peer_data))
            figs.append(make_sinr_comparison_plot(peer_data))
            figs.append(make_band_occupancy_evolution(peer_data))
    except Exception as e:
        print(f"[WARN] Failed to create aggregate plots: {e}")
    
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pdf_path = os.path.join(DATA_DIR, f"{node_id}_plots_{timestamp}.pdf")
    save_plots_to_pdf(pdf_path, figs)
    
    with open(os.path.join(DATA_DIR, f"{node_id}_log_{timestamp}.txt"), "w") as f:
        f.write(f"=== RECEIVER {node_id} SESSION ===\n\n")
        for ev in base_station_events: f.write(ev + "\n")
        f.write("\n=== RECEIVED MESSAGES ===\n")
        for peer, entries in peer_data.items():
            f.write(f"\nFrom {peer} ({len(entries)} messages):\n")
            for e in entries: 
                status = "JAMMED" if "[JAMMED]" in e["text"] else "CLEAN"
                f.write(f"  [{status}] {e['text']} (QAM-{e['M']})\n")
    print(f"[EXPORT] Saved {len(figs)} comprehensive receiver plots to {pdf_path}")

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