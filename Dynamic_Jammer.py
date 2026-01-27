# Dynamic_Jammer.py - MALICIOUS INTERFERENCE NODE
# - Simulates a High-Power Barrage Jammer
# - Floods Base Station to trigger "Interference State"
# - Use this to test robustness of Sender/Receiver

import socket, time, random, json, struct, threading

# Target Base Station (Default BS1)
BS_HOST, BS_PORT = "127.0.0.1", 50050
JAMMER_ID = "JAMMER_01"

# Jamming Config
JAMMING_POWER = 1e-8  # Very high power (overpowers legitimate signals)
BURST_DURATION = 5.0  # Seconds to jam continuously
COOLDOWN = 5.0        # Seconds to sleep (Intermittent Jamming)

def main():
    print(f"\n[JAMMER] Initializing High-Power Jamming Node {JAMMER_ID}...")
    try:
        sock = socket.create_connection((BS_HOST, BS_PORT))
        # Register as a special "jammer" type
        reg_msg = {"type": "jammer", "id": JAMMER_ID, "pos": [50.0, 50.0]}
        sock.sendall((json.dumps(reg_msg) + "\n").encode("utf-8"))
        print(f"[JAMMER] Connected to Base Station. Starting Attack Cycle.")
    except Exception as e:
        print(f"[ERROR] Could not connect to BS: {e}")
        return

    try:
        while True:
            # 1. JAMMING PHASE
            print(f"\n[ATTACK] >>> BROADCASTING WIDEBAND NOISE ({BURST_DURATION}s) <<<")
            end_time = time.time() + BURST_DURATION
            
            while time.time() < end_time:
                # Send "Noise" packets to the BS
                # These act as a signal to the BS that the channel is corrupted
                noise_payload = list(random.getrandbits(8) for _ in range(64))
                
                # Custom Packet Format for Jammer: [ID_LEN][ID][POWER][NOISE_BYTES]
                # We use a distinct format so BS handles it strictly as interference
                id_bytes = JAMMER_ID.encode("utf-8")
                payload = struct.pack(">H", len(id_bytes)) + id_bytes + \
                          struct.pack(">d", JAMMING_POWER) + \
                          bytes(noise_payload)
                
                wire = struct.pack(">I", len(payload)) + payload
                try:
                    sock.sendall(wire)
                except:
                    break
                
                # Flood rate control (simulate continuous wave)
                time.sleep(0.1) 
            
            print(f"[ATTACK] Cooldown ({COOLDOWN}s)... Channel may recover.")
            
            # 2. COOLDOWN PHASE
            time.sleep(COOLDOWN)

    except KeyboardInterrupt:
        print("\n[JAMMER] Attack stopped manually.")
    except Exception as e:
        print(f"\n[JAMMER] Error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    main()