# Cognitive Radio Network (CRN) Simulation with Jamming & Spectrum Sensing

## Project Overview
This project is a high-fidelity Python simulation of a **Cognitive Radio Network (CRN)**. It models the physical (PHY) and network layers to demonstrate core CRN concepts: **Spectrum Sensing (Listen Before Talk)**, **Primary User Priority**, and **Resilience against Jamming Attacks**.

The system simulates a real-world wireless environment where "Secondary Users" (unlicensed) must opportunistically access the spectrum without interfering with "Primary Users" (licensed). It also includes a malicious "Jammer" node to test the network's robustness and security protocols.

## Key Features
* **Dynamic Spectrum Access:** Secondary users employ **Markov Chain-based Spectrum Sensing** to detect channel occupancy before transmitting. If the noise floor exceeds the threshold, the node backs off.
* **Primary User Protection:** The Base Station enforces strict priority. If a Primary User (e.g., JAZZ, UFONE) transmits, active Secondary users are preempted (queued or disconnected) to ensure licensed quality of service.
* **Physical Layer Simulation:**
    * **Modulation:** Adaptive modulation schemes including QAM-16, QAM-64, and QAM-256 combined with OFDM (Orthogonal Frequency-Division Multiplexing) for realistic signal modeling.
    * **Error Correction:** Reed-Solomon (RS) coding (RSCodec-40) to detect and correct transmission errors.
    * **Channel Modeling:** Simulates AWGN (Additive White Gaussian Noise) and variable interference levels.
* **Cybersecurity:**
    * **Encryption:** AES-GCM (Galois/Counter Mode) provides authenticated encryption, ensuring both confidentiality and data integrity.
    * **Key Derivation:** PBKDF2 (Password-Based Key Derivation Function 2) with salting is used to derive strong cryptographic keys from passphrases.
* **Jamming Simulation:** A specialized "Jammer" node that floods the Base Station with high-power noise. This triggers a **Bit-Flipping Attack** on legitimate packets, simulating physical layer corruption and Denial of Service (DoS).
* **Rich Visualization:** Generates detailed PDF reports containing Constellation Diagrams (visualizing signal quality), OFDM I/Q Signals, and Power Sensing plots.

## System Architecture

### 1. The Sender (Primary/Secondary)
The sender initiates communication by performing **Spectrum Sensing**. 
* **Sensing:** It checks the environmental noise floor. If below the threshold, it proceeds.
* **Processing:** The message is encoded (Reed-Solomon), modulated (QAM/OFDM), and encrypted (AES-GCM).
* **Transmission:** The secure frame is sent to the Base Station.

### 2. The Base Station (Controller)
Acts as the central relay and policy enforcer.
* **Jamming Check:** Continually monitors for high-power interference from registered Jammers. If detected, it applies a **Bit-Flipping** corruption algorithm to passing frames.
* **Priority Check:** Validates if the source is a Primary User. If yes, it updates the "Protection Window". If a Secondary User attempts to transmit during this window, the packet is queued or dropped.
* **Routing:** Forwards valid, allowed frames to the destination Receiver.

### 3. The Receiver (Target)
The final destination for data.
* **Preemption:** Checks incoming connection requests. If a Primary User connects while a Secondary User is active, the Receiver automatically terminates the Secondary session.
* **Decryption & Visualization:** Attempts to decrypt the payload. If successful, it logs the message. If decryption fails (due to jamming), it logs the corruption. Finally, it generates visual plots of the signal constellation.

## File Structure

| File | Role | Description |
| :--- | :--- | :--- |
| `base_station.py` | **The Core / Air Interface** | Acts as the central hub. Enforces priority logic, manages routing, and simulates physical layer corruption when jamming is active. |
| `Dynamic_Sender.py` | **Transmitter Node** | Can act as a **Primary** or **Secondary** user. Implements "Listen Before Talk" sensing logic, adaptive modulation, and encryption. |
| `Dynamic_Receiver.py` | **Target Node** | Receives frames, handles decryption, and generates the PDF visualizations. Includes logic to auto-disconnect Secondary users if a Primary user connects. |
| `Dynamic_Jammer.py` | **Attacker Node** | Broadcasts high-power noise to the Base Station to disrupt communication and trigger packet corruption. |

## Installation & Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install numpy matplotlib pycryptodome reedsolo
