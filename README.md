# Changes/Notes

- Implement multi channel communication
- fix receiver's replying to itself problem 

> Formula for energy: np.mean(np.abs(ofdm_sig)**2) if ofdm_sig.size>0 else 0.0 ===> Computing avg power=2.0302e-02 in the R2 - Spectrum Sensing (0) [Page 3]