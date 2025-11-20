# Changes/Notes

- Check if Listen Before Talk approach is even viable or not [Yes, Research Paper Found]
- add queuing for secondary user instead of retrying again once every 2 seconds
- threading [allow sender instances to communicate with one receiver instance]
- fix receiver's replying to itself problem 


> energy = np.mean(np.abs(ofdm_sig)**2) if ofdm_sig.size>0 else 0.0 ===> Computing avg power=2.0302e-02 in the R2 - Spectrum Sensing (0) [Page 3]


* in ofdm and 5g communication pipeline, can there be multiple channels in a single base station such that there can be multiple recievers and senders communicating at the same time?
If yes, then how can we implement this in the current codebase