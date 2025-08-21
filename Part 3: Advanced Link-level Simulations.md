# Part 3: Advanced Link-level Simulations
本教學將引導你從 Sionna 的基本原則出發  
* 實作出一個符合 5G NR 標準編碼
* 使用 3GPP 通道模型的 point-to-point 通信鏈路
你也將學會如何撰寫自定義的 trainable blocks，用 TensorFlow 定義自己的神經網路模組（例如用 CNN、Transformer 當接收器）   
* 透過實作最先進的神經網路接收器，來訓練與評估整體的端到端通信系統

***

## 本篇內容主要分成五大章節
1. Imports
2. OFDM Resource Grid and Stream Management
3. Antenna Arrays
4. Channel Model
5. Uplink Transmission in the Frequency Domain

***

## Imports
```python
import os # Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna as sn
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np

# For plotting
%matplotlib inline
import matplotlib.pyplot as plt

# For the implementation of the Keras models
from tensorflow.keras import Model

# Set seed for reproducable results
sn.phy.config.seed = 42
```

***

## OFDM Resource Grid and Stream Management
接著要在 User 和 Basestation 之間建立一條 SIMO point-to-point link ，系統流程如下  
<img width="556" height="243" alt="image" src="https://github.com/user-attachments/assets/c6f62071-5843-4cb3-aa08-25689a7081e5" />

***

### Stream Management
* 不管哪種類型的 MIMO system ，我們都必須建立一個 `StreamManagement` 物件   
* 這個物件會決定**哪個TX會和哪個RX互相傳輸data stream**
  
在本次範例的系統中，配有一個單天線 UE 和一個多天線 BS
* 誰擔任TX、RX 都可以被設定，取決於你想模擬 uplink or downlink
* 補充: 為了滿足 spatial multiplexing 每個 TX 的 data stream 數量必須小於等於 UE 天線數量

```python
# 定義UE,BS參數
NUM_UT = 1
NUM_BS = 1
NUM_UT_ANT = 1
NUM_BS_ANT = 4

NUM_STREAMS_PER_TX = NUM_UT_ANT
RX_TX_ASSOCIATION = np.array([[1]])
STREAM_MANAGEMENT = sn.phy.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)
```
* RX_TX_ASSOCIATION: RX-TX 之間的相關矩陣。若矩陣的[i,j]=1，表示接收器 i 至少從發射器 j 取得一個 data stream
* 在本次範例中，因為 TX,RX 的數量都只有1個，因此該矩陣為 np.array([[1]])
* 對於更複雜的多收多發的系統，它可能長這樣
```python
RX_TX_ASSOCIATION = np.array([
    [1, 1, 0, 0],  # receiver 0 接收 TX 0 和 TX 1 的資料
    [0, 0, 1, 1]   # receiver 1 接收 TX 2 和 TX 3 的資料
])
```
* `sn.phy.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)`: 根據前面設置的 `RX_TX_ASSOCIATE`, `NUM_STREAMS_PER_TX`，設置成一組物件

***

### OFDM Resource Grid
這裡的 ResourceGrid 是一個二維資源網格，用來模擬 OFDM 系統中「時間-頻率」的資源分配  
* 時間軸 → 不同的 OFDM symbol（例如 14 個 symbol 形成 1 slot）
* 頻率軸 → 不同的子載波（由 `fft_size` 控制）  

這個 ResourceGrid 的結構包含:
* data symbol）
* pilot symbols / reference signal
* guard subcarriers: 頻譜兩側空出保護頻帶，避免干擾
* DC subcarrier : 5G NR 會將 DC subcarrier 設為0

範例程式碼
```python
RESOURCE_GRID = sn.phy.ofdm.ResourceGrid(num_ofdm_symbols=14,                      # 模擬 14 個 OFDM symbol
                                         fft_size=76,                              # subcarrier 數量  
                                         subcarrier_spacing=30e3,                  # subcarrier 的頻寬
                                         num_tx=NUM_UT,                            # TX 數量
                                         num_streams_per_tx=NUM_STREAMS_PER_TX,    # 每個 TX 要傳幾個 stream
                                         cyclic_prefix_length=6,                   # CP 長度 (單位為 FFT sample 數)
                                         pilot_pattern="kronecker",                # 使用 Kronecker pilot pattern 
                                         pilot_ofdm_symbol_indices=[2,11])         # 在第 2 與第 11 個 OFDM symbol 插入 pilot OFDM symbol
RESOURCE_GRID.show();   # 繪製上述的 OFDM ResourceGrid               
```
<img width="606" height="440" alt="image" src="https://github.com/user-attachments/assets/09ca86d9-4b66-49c9-bf99-22e025bede77" />

***

## Antenna Arrays
在 Sionna 中進行實體層（PHY layer）模擬時:  
* 如果你選擇的是簡單的通道模型（例如 AWGN、Rayleigh、TDL）， 這些模型不會考慮**天線幾何與波束特性**，因此 可以**不用設定天線陣列**
* 但在實際情況中，如果使用真實世界的通道模型 (像是3GPP 38.901 規格中的):
    * CDL（Clustered Delay Line）
    * UMi（Urban Micro）
    * UMa（Urban Macro）
    * RMa（Rural Macro）
  此時就**須配置天線陣列**

範例程式碼
```python
# 為確保天線之間的獨立，各個天線元件在垂直與水平方向的間距為半個波長（λ/2)
CARRIER_FREQUENCY = 2.6e9                                                            # 載波頻率，會影響到天線間隔
                                            
UT_ARRAY = sn.phy.channel.tr38901.Antenna(polarization="single",                     # 單極化天線
                                          polarization_type="V",                     # 垂直極化
                                          antenna_pattern="38.901",                  # 使用3GPP 38.901標準定義的天線陣列
                                          carrier_frequency=CARRIER_FREQUENCY)       # 根據子載波頻率決定天線間距     

BS_ARRAY = sn.phy.channel.tr38901.AntennaArray(num_rows=1,                           # 天線陣列 row 
                                               num_cols=int(NUM_BS_ANT/2),           # 天線陣列 column 
                                               polarization="dual",                  # 雙極化
                                               polarization_type="cross",            # 交叉極化
                                               antenna_pattern="38.901",             # 使用3GPP 38.901標準定義的天線陣列 
                                               carrier_frequency=CARRIER_FREQUENCY)  # 根據子載波頻率決定天線間距
UT_ARRAY.show();
BS_ARRAY.show();
```
<img width="595" height="908" alt="image" src="https://github.com/user-attachments/assets/530b6a7b-8aa1-413a-a0bc-bc04078177a8" />  

* 上圖為 UT_ARRAY: 對應到單一天線垂直極化
* 下圖為 BS_ARRAY: 陣列上有兩個物理位置；每個位置上放置兩根互相垂直的天線，所以天線數量=4。並且間距為5.77cm=λ/2

***

## Channel Model
Sionna 支援多種通道模型，這些模型來自 3GPP TR 38.901 標準：  
* CDL（Clustered Delay Line）
* TDL（Tapped Delay Line）
* UMi（Urban Micro）
* UMa（Urban Macro）
* RMa（Rural Macro）
* 此外還支援 Rayleigh block fading，這是較簡化的隨機通道模型    

這些模型模擬了不同場景下的通道衰減、角度分佈、時延擴散、移動速度等無線通道特性，能幫助建立現實世界中更真實的 MIMO 傳輸模擬。









