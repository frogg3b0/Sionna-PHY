# Sionna-PHY
## Introduction to Sionna PHY
### 此套件提供一個可微分的鏈路層模擬器（link-level simulator）  
* link-level 模擬器: 是指針對發送端與接收端之間的點對點通訊鏈路進行模擬
* 可微分: 表示這個模擬器的各個模組都支援 TensorFlow 的自動微分，這樣才能串接 AI 模型進行訓練，以及實現端對端學習

***

### 此外，Sionna PHY還無縫整合多個通訊系統元件，例如:  
* 前向錯誤更正（FEC）
* 符合 5G NR 規範的 encoder/decoder
* 多輸入多輸出（MIMO）系統
* 正交分頻多工（OFDM）
* 符合 3GPP 規範的無線通訊通道模型

***

### 給初學者
* Part 0: "Hellow world!"
* Part 1: Getting Started with Sionna
* Part 2: Differentiable Communication Systems
* Part 3: Advanced Link-level Simulations
* Part 4: Toward Learned Receivers
* Part 5: Basic MIMO Simulations
* Part 6: Pulse-shaping Basics
* Part 7: Optical Channel with Lumped Amplification

### 給進階開發者
進階使用者可參閱後續內容，以深入理解 Sionna PHY 的內部運作方式，並學習如何擴展與自訂物理層演算法。
* 5G Channel Coding and Rate-Matching: Polar vs. LDPC Codes
* 5G NR PUSCH Tutorial
* Bit-Interleaved Coded Modulation (BICM)
* MIMO OFDM Transmissions over the CDL Channel Model
* Neural Receiver for OFDM SIMO Systems
* Realistic Multiuser MIMO OFDM Simulations
* OFDM MIMO Channel Estimation and Detection
* Introduction to Iterative Detection and Decoding
* End-to-end Learning with Autoencoders
* Weighted Belief Propagation Decoding
* Channel Models from Datasets
* Using the DeepMIMO Dataset with Sionna
* Link-level simulations with Sionna RT

***

### Sionna PYH 實作範例
#### 範例1: QAM symbols over an AWGN channel (Part 1)
<img width="656" height="692" alt="image" src="https://github.com/user-attachments/assets/9fd23b85-f429-4e0d-8456-c12e9889ab95" />  

***

#### 範例2: 使用 5G Polor/LDCP 的 encoder/decoder (Part 1)
<img width="622" height="165" alt="image" src="https://github.com/user-attachments/assets/241a089f-b0ab-4ef5-9cfa-16e63d44f09e" />   
<img width="651" height="424" alt="image" src="https://github.com/user-attachments/assets/e5b5b3a3-e2d6-4cf3-9cdb-33255d57dfca" />  

***

### 範例3: OFDM Resource Grid (Part 3)
<img width="563" height="240" alt="image" src="https://github.com/user-attachments/assets/0e245ef2-f02c-4859-b439-c75398ef4d02" />  
<img width="607" height="438" alt="image" src="https://github.com/user-attachments/assets/efc460fd-af96-41ae-a4cd-9d75f063063e" />  

***

### 範例4: 使用 3GPP model (Part 3)
<img width="563" height="243" alt="image" src="https://github.com/user-attachments/assets/6d1b4a7d-38fa-448a-a9f0-f1d4503eb2ab" />  
<img width="584" height="450" alt="image" src="https://github.com/user-attachments/assets/bf4abb68-e7f9-44b3-89fb-e5ca029ac7cd" />  
<img width="650" height="430" alt="image" src="https://github.com/user-attachments/assets/0fadb0ad-1075-49af-83d4-fd832ac3550d" />


***

### 範例5: Implemention of an Advanced Neural Receiver (Part 4)
<img width="677" height="162" alt="image" src="https://github.com/user-attachments/assets/fef2b27d-fe5f-475a-a9f1-742e380da3ac" />
<img width="694" height="602" alt="image" src="https://github.com/user-attachments/assets/afcdd2bc-b72d-4b0e-8917-42b78a43c1ba" /> 
<img width="669" height="428" alt="image" src="https://github.com/user-attachments/assets/c3792799-8b51-4a6a-969b-c318f6a286bc" />









