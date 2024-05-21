# GFANC-RL

This repository contains the code for the paper "**GFANC-RL: Reinforcement Learning-based Generative Fixed-filter Active Noise Control**," submitted to the **Neural Networks** journal. This is a collaborative research work between the Digital Signal Processing Lab at NTU and the AI Lab at NUS. The complete code will be made publicly available once the paper is accepted.

<p align="center">
  <img src="https://github.com/Luo-Zhengding/GFANC-RL/assets/95018034/07d44cdd-b60a-44b4-b1c5-6442d925b7f4" width="500"><br>
  The framework of the GFANC-RL method
</p>

<p align="center">
  <img src="https://github.com/Luo-Zhengding/GFANC-RL/assets/95018034/c44ad8c5-dafb-4811-b169-fb3ebdafd9ec" width="300">
  <img src="https://github.com/Luo-Zhengding/GFANC-RL/assets/95018034/04d1ab12-beb7-4123-b680-4cf8b91d3173" width="300"><br>
  Parameter update for the critic in the RL algorithm. &nbsp; &nbsp; &nbsp; Parameter update for the actor in the RL algorithm.
</p>

## Highlights
1. GFANC-RL employs RL techniques to address challenges associated with GFANC innovatively.
2. This paper formulates the GFANC problem as a Markov Decision Process (MDP) from a decision-making perspective, laying a theoretical foundation for using RL algorithms.
3. In the GFANC-RL method, an RL algorithm based on Soft Actor-Critic (SAC) is developed to train the CNN using unlabelled noise data and improve the exploration ability of the CNN model.
4. Experimental results show that the GFANC-RL method effectively attenuates real-recorded noises and exhibits good robustness and transferability in different acoustic paths.

## Usage
### 1. Obtaining sub control filters
- The optimal control filter of broadband noise, whose frequency band contains our interested components, is first chosen as the full-band response. Subsequently, it is decomposed into M orthogonal sub bands as the desired sub control filters.
  <p align="center">
  <img src="https://github.com/Luo-Zhengding/GFANC-RL/assets/95018034/3e6c9c78-b194-42c5-bb15-427e04b6a0d7" width="300">
  </p>


### 2. Training the CNN using RL
- A synthetic noise dataset is used to train the CNN, containing 80,000 noise instances for training. The noise instances are generated by filtering white noise through various bandpass filters with randomly chosen center frequencies and bandwidths. Each noise instance has a 1-second duration. The noise dataset is available at - [Training dataset](https://drive.google.com/file/d/1hs7_eHITxL16HeugjQoqYFTs-Cm7J-Tq/view?pli=1)
- It is worth noting that no data labels are used in the training phase.

### 3. Real-time Noise Cancellation
- After training via the RL algorithm, the GFANC-RL method is used for real-time noise control.

### Applying to New Environments
- Transferring the GFANC-RL method to new systems involves only updating the system-specific sub control filters, with the trained 1D CNN remaining unchanged, thus simplifying implementation across various scenarios.

## Related Works
- [Delayless Generative Fixed-filter Active Noise Control based on Deep Learning and Bayesian Filter](https://ieeexplore.ieee.org/document/10339836/)
- [Deep Generative Fixed-Filter Active Noise Control](https://arxiv.org/pdf/2303.05788)
- [GFANC-Kalman: Generative Fixed-Filter Active Noise Control with CNN-Kalman Filtering](https://ieeexplore.ieee.org/document/10323505)
- [Real-time implementation and explainable AI analysis of delayless CNN-based selective fixed-filter active noise control](https://www.sciencedirect.com/science/article/abs/pii/S0888327024002620)
- [A hybrid sfanc-fxnlms algorithm for active noise control based on deep learning](https://arxiv.org/pdf/2208.08082)
- [Performance Evaluation of Selective Fixed-filter Active Noise Control based on Different Convolutional Neural Networks](https://arxiv.org/pdf/2208.08440)

**If you are interested in our works, please consider citing our papers. Thanks! Have a great day!**
