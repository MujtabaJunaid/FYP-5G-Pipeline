# base_station.py (Optimized + Primary Uplink + JAMMING SCENARIO)
# - Includes Jamming/Interference Simulation
# - Corrupts packets when Jammer is active
# - Maintains all existing Primary/Secondary logic

import socket, threading, json, struct, time, os, math, glob, hashlib, random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reedsolo import RSCodec
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
import numpy as np

# --- CONFIGURATION ---
BS_INSTANCE = 1 
# ---------------------

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
HOST = "127.0.0.1"
COMM_RANGE = 150.0

if BS_INSTANCE == 1:
    PORT = 50050; BS_ID = "BS1"; BS_POS = (0.0, 0.0)
    NEIGHBORS = [("127.0.0.1", 50051)]
else:
    PORT = 50051; BS_ID = "BS2"; BS_POS = (200.0, 0.0)
    NEIGHBORS = [("127.0.0.1", 50050)]

# ---------- Crypto ----------
PASSPHRASE = b"very secret passphrase - change this!"
SALT = b'\x00'*16 
KEY = PBKDF2(PASSPHRASE, SALT, dkLen=32, count=100000)
rs = RSCodec(40)

# ---------- State ----------
clients_lock = threading.Lock()
clients = {}
retry_queue = deque()
retry_queue_lock = threading.Lock()
stats = { "received": 0, "forwarded_local": 0, "forwarded_remote": 0, "queued": 0, "delivered": 0, "jammed": 0 }
stats_lock = threading.Lock()
STOP = threading.Event()

FRAME_PROCESSOR_POOL = ThreadPoolExecutor(max_workers=10)

PRIMARY_SENDERS = ["JAZZ", "UFONE", "TELENOR", "WARID", "STARLINK", "ZONG", "SCO", "PTCL"]
ENERGY_HISTORY = []
LAST_PRIMARY_ACTIVITY = 0.0
PRIMARY_PROTECTION_WINDOW = 10.0

# --- JAMMING STATE ---
JAMMING_ACTIVE = False
JAMMING_LAST_SEEN = 0.0
JAMMING_TIMEOUT = 0.5 # Reset jamming state if no jammer signal for 0.5s

def distance(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def recv_full(sock, length):
    data = b""
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more: raise ConnectionError("closed")
        data += more
    return data

def send_all_with_retry(conn, data):
    try: conn.sendall(data); return True
    except: return False

# PHY Helpers
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

def process_incoming_frame(src_id, dst_id, encoded_bytes, hop_count=0):
    global JAMMING_ACTIVE, JAMMING_LAST_SEEN
    
    # 0. JAMMING CHECK (Physical Layer Corruption Simulation)
    # If a jammer was recently seen, we corrupt the packet
    if time.time() - JAMMING_LAST_SEEN < JAMMING_TIMEOUT:
        JAMMING_ACTIVE = True
    else:
        JAMMING_ACTIVE = False

    final_payload = encoded_bytes
    if JAMMING_ACTIVE:
        print(f"[BS {BS_ID}] ! JAMMING ACTIVE ! Corrupting packet from {src_id}...")
        with stats_lock: stats["jammed"] += 1
        
        # --- BIT FLIPPING ATTACK (Simulate Noise) ---
        # We turn the bytes into a mutable bytearray and flip random bits
        # This will cause RS-Decoding or AES-GCM tag verification to fail at Receiver
        ba = bytearray(encoded_bytes)
        # Corrupt 30% of bytes to simulate heavy interference
        corruption_intensity = int(len(ba) * 0.3) 
        for _ in range(corruption_intensity):
            idx = random.randint(0, len(ba)-1)
            ba[idx] = ba[idx] ^ random.randint(1, 255) # XOR with noise
        final_payload = bytes(ba)
        # --------------------------------------------

    try:
        # 1. Sensing & Decoding (Try to decode even if corrupted - might fail now)
        try:
            payload_to_decode = final_payload[8:] if len(final_payload) >= 8 else final_payload
            rs_decoded = rs.decode(payload_to_decode)[0]
            plaintext = aes_gcm_decrypt(rs_decoded, KEY)
            M, nc, cp = struct.unpack(">H H B", plaintext[:5])
            msg_bytes = plaintext[5:][2+struct.unpack(">H", plaintext[5:][:2])[0]:]
            
            tx_syms = qam_mod(bits_from_bytes(msg_bytes), M)
            ofdm_sig = ofdm_mod(tx_syms, nc, cp)
            energy = float(np.mean(np.abs(ofdm_sig)**2)) if ofdm_sig.size > 0 else 0.0
            
            ENERGY_HISTORY.append((time.time(), src_id, energy))
            
            if src_id.upper() in PRIMARY_SENDERS:
                global LAST_PRIMARY_ACTIVITY
                LAST_PRIMARY_ACTIVITY = time.time()
                print(f"[BS {BS_ID}] Primary Activity: {src_id} (E={energy:.2e})")
        except: 
            # If decoding failed (likely due to Jamming), we just pass.
            pass

        # 2. Priority Logic
        is_primary_link = (src_id.upper() in PRIMARY_SENDERS) or (dst_id.upper() in PRIMARY_SENDERS)
        
        if (not is_primary_link) and (time.time() - LAST_PRIMARY_ACTIVITY < PRIMARY_PROTECTION_WINDOW):
            print(f"[BS {BS_ID}] Deferring Secondary {src_id}->{dst_id} (Channel protected)")
            with stats_lock: stats["queued"] += 1
            with retry_queue_lock:
                retry_queue.append({"src": src_id, "dst": dst_id, "data": final_payload, "hop": hop_count, "ts": time.time()})
            return

        # 3. Routing
        with clients_lock: dst_info = clients.get(dst_id)
        if dst_info and distance(BS_POS, dst_info["pos"]) <= COMM_RANGE:
            src_b = src_id.encode("utf-8")
            payload = struct.pack(">H", len(src_b)) + src_b + final_payload
            if send_all_with_retry(dst_info["conn"], struct.pack(">I", len(payload)) + payload):
                print(f"[BS {BS_ID}] Delivered: {src_id} -> {dst_id}")
                with stats_lock: stats["delivered"] += 1; stats["forwarded_local"] += 1
                return

        src_b, dst_b = src_id.encode("utf-8"), dst_id.encode("utf-8")
        hop_payload = struct.pack(">B H", hop_count+1, len(src_b)) + src_b + struct.pack(">H", len(dst_b)) + dst_b + final_payload
        for nb_h, nb_p in NEIGHBORS:
            try:
                with socket.create_connection((nb_h, nb_p), timeout=2) as s:
                    s.sendall(struct.pack(">I", len(hop_payload)) + hop_payload)
                    print(f"[BS {BS_ID}] Hopped: {src_id} -> {dst_id}")
                    with stats_lock: stats["forwarded_remote"] += 1
                    return
            except: continue

        print(f"[BS {BS_ID}] Queueing {dst_id} (Unreachable)")
        with retry_queue_lock:
            retry_queue.append({"src": src_id, "dst": dst_id, "data": final_payload, "hop": hop_count, "ts": time.time()})
            
    except Exception as e: print(f"[BS {BS_ID}] Error: {e}")

def handle_registered_client(conn, addr):
    try:
        buf = b""
        while b"\n" not in buf: buf += conn.recv(1024)
        line, buf = buf.split(b"\n", 1)
        reg = json.loads(line.decode("utf-8"))
        client_id = reg["id"]
        client_type = reg.get("type", "sender")
        
        # --- JAMMER HANDLING ---
        if client_type == "jammer":
            print(f"[BS {BS_ID}] !!! WARNING: JAMMER CONNECTED ({client_id}) !!!")
            # Loop specifically for Jammer traffic to avoid processing overhead
            while not STOP.is_set():
                # Read Jammer packet headers [ID_LEN (2)]
                h = recv_full(conn, 4) 
                # Just consume the data stream to "sense" the jamming
                length = struct.unpack(">I", h)[0]
                _ = recv_full(conn, length)
                
                # Update Jamming State
                global JAMMING_LAST_SEEN
                JAMMING_LAST_SEEN = time.time()
                # We don't forward jammer packets, we just register the interference
                # print(f"[BS {BS_ID}] Interference detected from {client_id}")
            return
        # -----------------------

        with clients_lock: clients[client_id] = {"conn": conn, "pos": tuple(reg["pos"])}
        print(f"[BS {BS_ID}] Registered {client_id}")

        while not STOP.is_set():
            while len(buf) < 4: buf += conn.recv(4096)
            length, buf = struct.unpack(">I", buf[:4])[0], buf[4:]
            while len(buf) < length: buf += conn.recv(4096)
            payload, buf = buf[:length], buf[length:]
            dst_len = struct.unpack(">H", payload[:2])[0]
            dst_id = payload[2:2+dst_len].decode("utf-8")
            remaining = payload[2+dst_len:]
            with stats_lock: stats["received"] += 1
            FRAME_PROCESSOR_POOL.submit(process_incoming_frame, client_id, dst_id, remaining)
    except: pass
    finally:
        with clients_lock: clients.pop(client_id, None) if 'client_id' in locals() else None
        conn.close()

def handle_hop_connection(conn, addr):
    try:
        payload = recv_full(conn, struct.unpack(">I", recv_full(conn, 4))[0])
        hop, pos = payload[0], 1
        src_len = struct.unpack(">H", payload[pos:pos+2])[0]; pos += 2
        src = payload[pos:pos+src_len].decode("utf-8"); pos += src_len
        dst_len = struct.unpack(">H", payload[pos:pos+2])[0]; pos += 2
        dst = payload[pos:pos+dst_len].decode("utf-8"); pos += dst_len
        FRAME_PROCESSOR_POOL.submit(process_incoming_frame, src, dst, payload[pos:], hop)
    except: pass
    finally: conn.close()

def retry_worker():
    while not STOP.is_set():
        time.sleep(1.0)
        if time.time() - LAST_PRIMARY_ACTIVITY < PRIMARY_PROTECTION_WINDOW: continue
        with retry_queue_lock:
            for _ in range(len(retry_queue)):
                item = retry_queue.popleft()
                FRAME_PROCESSOR_POOL.submit(process_incoming_frame, item["src"], item["dst"], item["data"], item["hop"])

def main():
    threading.Thread(target=retry_worker, daemon=True).start()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, PORT))
        sock.listen(16)
        print(f"[BS {BS_ID}] Listening on {PORT}")
        try:
            while not STOP.is_set():
                conn, addr = sock.accept()
                if conn.recv(1, socket.MSG_PEEK) == b'{': threading.Thread(target=handle_registered_client, args=(conn, addr), daemon=True).start()
                else: threading.Thread(target=handle_hop_connection, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt: STOP.set(); FRAME_PROCESSOR_POOL.shutdown(wait=False)

if __name__ == "__main__":
    main()