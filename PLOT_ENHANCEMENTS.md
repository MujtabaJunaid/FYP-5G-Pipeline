# Comprehensive Plotting Enhancements

## Overview
Enhanced all three main components (Sender, Receiver, Base Station) with elaborate, multi-layered visualization capabilities including before/after jamming analysis, real-time metrics, and advanced PHY layer analytics.

---

## **SENDER (s.py) Enhancements** 📊

### NEW Plotting Functions:
1. **Power Spectral Density (PSD)** - FFT-based frequency domain analysis
2. **Energy Histogram** - Distribution of energy across all sent messages
3. **Modulation Comparison** - Bar chart of QAM scheme usage
4. **3D Symbol Scatter** - Magnitude and phase information in 3D
5. **Amplitude Histogram** - Symbol amplitude distribution
6. **Phase Distribution** - Phase angle histogram of modulated symbols
7. **SNR Estimate** - Signal-to-Noise Ratio analysis per message
8. **Message Timeline** - Temporal distribution with recipient tracking

### Per-Message Analysis:
- Constellation plot (I-Q diagram)
- OFDM time-domain signal visualization
- Power envelope with threshold overlay
- Power Spectral Density
- SNR estimates
- Amplitude statistics
- Phase statistics

### Aggregate Metrics:
- Energy distribution across all recipients
- Modulation scheme frequency analysis
- Transmission timeline with color-coded primary/secondary senders

---

## **RECEIVER (r.py) Enhancements** 📈

### NEW Plotting Functions:
1. **Received Energy Histogram** - Energy per peer with primary/secondary distinction
2. **Jammed vs Clean Comparison** - Packet corruption statistics
3. **Packets per Peer** - Message count distribution with sender priority
4. **3D Symbol Scatter** - Magnitude and phase of received symbols
5. **Amplitude Histogram** - Receive-side amplitude analysis
6. **Phase Distribution** - Phase characteristics of received signals
7. **SNR Estimate** - Received signal quality analysis
8. **Packet Timeline** - Temporal sequence with jamming annotations
9. **Corruption Pattern** - Visual timeline of packet integrity status

### Per-Message Analysis:
- Clean/Jammed status indication (color-coded: blue=clean, red=jammed)
- Constellation plots with visual jamming indicators
- OFDM time-domain reconstruction
- Power envelope analysis
- PSD analysis
- SNR estimates
- Statistical distributions

### Aggregate Metrics:
- Total clean vs jammed packet count (bar chart with percentages)
- Received energy per sender (primary senders highlighted in red)
- Message distribution across peers
- Temporal packet timeline showing jamming events
- Corruption pattern visualization showing packet integrity over time

---

## **BASE STATION (b1.py) Enhancements** ⚙️

### NEW Analysis Functions:
1. **Energy Timeline** - Scatter plot of signal energy over time (color-coded by sender type)
2. **Energy Histogram** - Average energy per source
3. **Jamming Activity Timeline** - Detection markers for jammer interference events
4. **Statistics Summary** - Bar chart of all operational metrics
5. **Primary vs Secondary Distribution** - Pie chart of traffic composition
6. **Delivery Metrics** - Summary boxes for key performance indicators

### Generated Artifacts:
- **PDF Report**: `{BS_ID}_analysis_{timestamp}.pdf`
  - 6+ comprehensive plots automatically generated on shutdown
  - Energy analysis across all sources
  - Jamming detection timeline
  - Performance statistics
  
- **Statistics Log**: `{BS_ID}_stats_{timestamp}.txt`
  - Detailed metrics breakdown
  - Top senders list
  - Jamming event timestamps
  - Traffic statistics

---

## **Before/After Jamming Analysis** 🔴

### Cross-File Jamming Detection:
- **Sender perspective**: Individual packet analysis shows energy/BER before jamming
- **Receiver perspective**: Visual distinction of corrupted vs clean packets
  - Clean packets: Blue constellation, normal amplitude
  - Jammed packets: Red constellation with added noise (visual simulation)
- **Base Station perspective**: Jamming timeline shows interference event detection

### Jamming Impact Visualization:
- Packet corruption rate over time
- Energy spike correlation with jammer activity
- Primary sender protection effectiveness
- Queue buildup during jamming periods

---

## **Key Features Preserved** ✅

1. **Thread Safety**: All plot operations use `plot_lock` for thread-safe PDF generation
2. **Non-blocking UI**: Matplotlib Agg backend prevents editor/terminal freezes
3. **Existing Functionality**: All original communication logic intact
4. **Backward Compatibility**: Original simple plots still available in code
5. **Dynamic Thresholding**: Plots adapt to actual signal characteristics
6. **Error Handling**: Robust exception handling prevents plot failures from crashing main application

---

## **Output Files Generated**

### Sender Output:
```
node_logs/{NODE_ID}_plots_{timestamp}.pdf    (15-25 pages per session)
node_logs/{NODE_ID}_log_{timestamp}.txt      (detailed message log)
```

### Receiver Output:
```
node_logs/{NODE_ID}_plots_{timestamp}.pdf    (20-30 pages per session)
node_logs/{NODE_ID}_log_{timestamp}.txt      (jamming events + messages)
```

### Base Station Output:
```
results/{BS_ID}_analysis_{timestamp}.pdf     (6-7 analytical plots)
results/{BS_ID}_stats_{timestamp}.txt        (metrics breakdown)
```

---

## **Plot Statistics**

| Component | Plots per Message | Aggregate Plots | Total per Session |
|-----------|--|--|--|
| Sender    | 7-8 plots        | 3 plots         | 30-50 pages       |
| Receiver  | 7-8 plots        | 5 plots         | 40-60 pages       |
| BaseStation| N/A             | 6 plots         | 6-7 pages         |

---

## **Performance Optimization Notes**

- Signal limiting to first 300-1000 samples for complex plots to avoid rendering delays
- FFT computation only executed on finite signals
- Matplotlib figure pooling prevents memory leaks during extended sessions
- Thread-safe PDF generation with automatic cleanup

---

## **Testing Recommendations** 🧪

1. Send messages with varied recipient types (primary/secondary)
2. Activate jammer to see before/after corruption visualization
3. Check PDF outputs for plot completeness
4. Verify no terminal freezes during PDF generation
5. Confirm all statistics match actual traffic patterns

---

Generated: February 27, 2026
