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

## Channel Model
Sionna 支援多種通道模型，這些模型來自 [3GPP TR 38.901](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173) 標準：  
* CDL: 支援單一使用者（Single-User），但可以有多根天線（MIMO）
* TDL: 用於 SISO（單輸入單輸出）通道模擬，無法模擬 MIMO
* UMi: 支援單一使用者（Single-User），但可以有多根天線（MIMO）
* UMa: 支援單一使用者（Single-User），但可以有多根天線（MIMO）
* RMa: 支援單一使用者（Single-User），但可以有多根天線（MIMO）
* 此外還支援 Rayleigh block fading，這是較簡化的隨機通道模型    

這些模型模擬了不同場景下的通道衰減、角度分佈、時延擴散、移動速度等無線通道特性，能幫助建立現實世界中更真實的 MIMO 傳輸模擬。  
補充說明: TDL/CDL的每個 path都有固定的 power delay profile與固定的AoA / AoD。換句話說，它們是 deterministic，並非隨機生成  

### 首先考慮3GPP CDL model
point-to-point流程如下  
<img width="569" height="246" alt="image" src="https://github.com/user-attachments/assets/137692ab-d05f-4343-808a-5ea5a4fc56db" />  

```python
DELAY_SPREAD = 100e-9                               # 訊號因多路徑，到達接收端時間差異的分布              
DIRECTION = "uplink"                                # 決定誰是TX，誰是RX
CDL_MODEL = "C"                                     # CDL 模型族群中的 Model C 作為目前的通道設定
SPEED = 10.0                                        # 使用者UT的移動速度 [m/s]

# 根據上述參數，建構 CDL 通道模型
CDL = sn.phy.channel.tr38901.CDL(CDL_MODEL,         # 使用 3GPP 38.901 CDL model 
                                 DELAY_SPREAD,      # 輸入delay spread
                                CARRIER_FREQUENCY,  # 輸入載波頻率
                                UT_ARRAY,           # 指定使用 UT_ARRAY 做為使用者端的天線陣列
                                BS_ARRAY,           # 指定使用 BS_ARRAY 做為基地台端的天線陣列
                                DIRECTION,          # uplink/downlink
                                min_speed=SPEED)    # 用戶移動速度
```
建立好的 CDL 可以被用來隨機產生通道脈衝響應 (CIR)，包含  
* 每條路徑的複數增益 a
* 對應的 delay 𝜏

為了模擬 time-varying channel，我們會以某個取樣頻率對CIR進行多次取樣，通常會取 OFDM symbol 次，取樣流程舉例如下:  
1. 先在time-domain建立delay tap，每個tap間隔為 T
2. 所以我們在time-domain上就會有[0,T,2T...NT]這些N個離散的tap
3. 接著把原始的通道脈衝響應對應到最近的tap上
4. 像是: 0ns->0 th tap； 5ns->2 nd tap； 8ns -> 4th tap
5. 如果我們在建立delay tap的時候，把時間切分得越細(也就是取樣頻率越高)，就能降低量化誤差

在後面的程式碼段中就實際執行了這個 sampling 動作  
```python
BATCH_SIZE = 128 
a, tau = CDL(batch_size=BATCH_SIZE,
             num_time_steps=RESOURCE_GRID.num_ofdm_symbols,
             sampling_frequency=1/RESOURCE_GRID.ofdm_symbol_duration)
```

* 在「觀察的時間區間」內，延遲 𝜏𝑙 被視為固定不變，也就是說，每一條路徑的 delay 是定值
* 隨時間改變的是：每一條路徑的複數衰減係數 𝑎𝑙(𝑡)，它是 time-varying

#### <在某個時間點t=定值，其對應的多條路徑CIR，即 ℎ(𝑡=定值, 𝜏)>
<img width="591" height="451" alt="image" src="https://github.com/user-attachments/assets/9c7c1f4f-6230-4064-9c5b-1a719cba450a" />  

#### <某一條路徑𝜏=定值，隨著不同時間點的gain，即 ℎ(𝑡, 𝜏=定值)>
<img width="571" height="448" alt="image" src="https://github.com/user-attachments/assets/5adc61c2-d89b-4ffe-974f-57f16b8e447f" />

***

## Uplink Transmission in the Frequency Domain
本章節開始進行「上行鏈路的通訊模擬」，特別是在frequency domain  
為了這樣的建模，做出一個假設：  
* 在每一個 OFDM symbol 的期間內，通道保持不變（quasi-static）。
* 因此，不會模擬到因通道變化導致的 子載波間干擾（ICI)

使用 `OFDMChannel` 模組，來執行 frequency domain 的通道模擬，這個模組的功能包含:  
* 對通道脈衝取樣
* 計算頻遇響應: 將 CIR 做 Fourier transform
* 將通道響應H(f)乘上輸入訊號X(f)

使用`TimeChannel`模組，來執行 time domain 的通道模擬，這個模組可以計算
* Doppler
* ICI
* CP

首先，模擬上行鏈路通訊的過程  
程式範例
```python
NUM_BITS_PER_SYMBOL = 2 # QPSK
CODERATE = 0.5

n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL) 
k = int(n*CODERATE)

binary_source = sn.phy.mapping.BinarySource()
encoder = sn.phy.fec.ldpc.LDPC5GEncoder(k, n)
mapper = sn.phy.mapping.Mapper("qam", NUM_BITS_PER_SYMBOL)
rg_mapper = sn.phy.ofdm.ResourceGridMapper(RESOURCE_GRID)
channel = sn.phy.channel.OFDMChannel(CDL, RESOURCE_GRID, add_awgn=True, normalize_channel=True, return_channel=True)
ls_est = sn.phy.ofdm.LSChannelEstimator(RESOURCE_GRID, interpolation_type="nn")
lmmse_equ = sn.phy.ofdm.LMMSEEqualizer(RESOURCE_GRID, STREAM_MANAGEMENT)
demapper = sn.phy.mapping.Demapper("app", "qam", NUM_BITS_PER_SYMBOL)
decoder = sn.phy.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)
```
* `n`: 整個 OFDM Resource Grid裡，一共可以傳送多少**編碼後的bits**
    * `RESOURCE_GRID.num_data_symbols`: OFDM Resource Grid 中，所有可用來放資料的**符號總數**
    * `NUM_BITS_PER_SYMBOL`
* `k`: 原始資料的bits數，會根據通道碼的碼率`CODERATE`去反推可以傳多少bits
* `binary_source`: 建立隨機資料產生器，產出 0/1 二位元 bit
* `encoder`: 建立 LDPC 通道編碼器，將 k 個資訊 bits 編碼成 n 個 冗餘保護的 bits
* `mapper`: 將每組 `NUM_BITS_PER_SYMBOL` bits 映射成一個 QAM symbol
* `rg_mapper`: 將所有 QAM symbols 依照 `RESOURCE_GRID` 中規定的位置 排入 OFDM 資源格子 (見本篇前半段內容)
* `channel`: 使用 CDL Channel model, 考慮AWGN, 對通道增益normalize, 回傳h(f)給receiver端做估計
* `ls_est`: 做 Least square estimation
* `lmmse_equ`: 使用 LMMSE（最小均方誤差） 進行通道等化
* `demapper`: 將 QAM symbol ➜ soft LLRs
* `decoder`: 解碼，接收 n 個 LLR回復出 k 個資訊 bits。 `hard_out=True` 表示輸出最終 0/1 決策

接著要模擬上行鏈路通訊傳輸過程，每個階段各層輸出的shape  
程式範例  
```python
no = sn.phy.utils.ebnodb2no(ebno_db=10.0,
                            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                            coderate=CODERATE,
                            resource_grid=RESOURCE_GRID)


bits = binary_source([BATCH_SIZE, NUM_UT, RESOURCE_GRID.num_streams_per_tx, k])
codewords = encoder(bits)
x = mapper(codewords)
x_rg = rg_mapper(x)

# Channel
y, h_freq = channel(x_rg, no)
print("Shape of y_rg: ", y.shape)
print("Shape of h_freq: ", h_freq.shape)

# Receiver
h_hat, err_var = ls_est (y, no)
x_hat, no_eff = lmmse_equ(y, h_hat, err_var, no)
llr = demapper(x_hat, no_eff)
bits_hat = decoder(llr)

print("Shape of bits: ", bits.shape)
print("Shape of codewords: ", codewords.shape)
print("Shape of x: ", x.shape)
print("Shape of x_rg: ", x_rg.shape)
print("Shape of h_hat: ", h_hat.shape)
print("Shape of err_var: ", err_var.shape)
print("Shape of x_hat: ", x_hat.shape)
print("Shape of no_eff: ", no_eff.shape)
print("Shape of llr: ", llr.shape)
print("Shape of bits_hat: ", bits_hat.shape)
```
輸出
```
Shape of bits:  (128, 1, 1, 912)  
Shape of codewords:  (128, 1, 1, 1824)  
Shape of x:  (128, 1, 1, 912)  
Shape of x_rg:  (128, 1, 1, 14, 76)  
Shape of y_rg:  (128, 1, 4, 14, 76)  
Shape of h_freq:  (128, 1, 4, 1, 1, 14, 76)  
Shape of h_hat:  (128, 1, 4, 1, 1, 14, 76)  
Shape of err_var:  (1, 1, 1, 1, 1, 14, 76)  
Shape of x_hat:  (128, 1, 1, 912)  
Shape of no_eff:  (128, 1, 1, 912)  
Shape of llr:  (128, 1, 1, 1824)  
Shape of bits_hat:  (128, 1, 1, 912)  
```

***

### 定義一個可重複模擬 OFDM 傳輸系統的模型類別
包含發送端、通道、接收端（含 LS 或 perfect CSI），並能在呼叫時自動模擬一整個 batch 的 bit 傳輸與解碼  
<img width="737" height="485" alt="image" src="https://github.com/user-attachments/assets/579031e0-5efd-434b-853d-47f37ab4d5a2" />  

```python
model = OFDMSystem(perfect_csi=True)
bits, bits_hat = model(batch_size=128, ebno_db=10.0)
```
只要用這行code就能快速模擬上圖的整套流程，他就會自動執行:  
1. 產生亂數位元
2. 做 LDPC 編碼 + QAM 映射 + OFDM 映射
3. 傳送過 CDL 通道 + AWGN
4. 使用 Perfect CSI 或 LS channel estimation
5. 做 LMMSE 等化、解調、解碼
6. 回傳原始 bit 與解碼 bit，計算 BER

實作程式碼細節

```python
class OFDMSystem(Model): # Inherits from Keras Model

    def __init__(self, perfect_csi):
        super().__init__() # Must call the Keras model initializer

        self.perfect_csi = perfect_csi

        n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL) # Number of coded bits
        k = int(n*CODERATE) # Number of information bits
        self.k = k
        # 發射端
        self.binary_source = sn.phy.mapping.BinarySource()                 # 產生隨機 bits
        self.encoder = sn.phy.fec.ldpc.LDPC5GEncoder(k, n)                 # LDPC 編碼，將 k bits → n bits
        self.mapper = sn.phy.mapping.Mapper("qam", NUM_BITS_PER_SYMBOL)    # 將編碼後的 bits 做 QAM 映射
        self.rg_mapper = sn.phy.ofdm.ResourceGridMapper(RESOURCE_GRID)     # 把符號排入 OFDM 網格
        # 接收端
        self.channel = sn.phy.channel.OFDMChannel(CDL, RESOURCE_GRID, add_awgn=True, normalize_channel=True, return_channel=True)
        self.ls_est = sn.phy.ofdm.LSChannelEstimator(RESOURCE_GRID, interpolation_type="nn")  # 用 pilot + interpolation 估計頻域通道
        self.lmmse_equ = sn.phy.ofdm.LMMSEEqualizer(RESOURCE_GRID, STREAM_MANAGEMENT)         # LMMSE Equalizer
        self.demapper = sn.phy.mapping.Demapper("app", "qam", NUM_BITS_PER_SYMBOL)            # 把符號轉回 LLR（log-likelihood ratio） 
        self.decoder = sn.phy.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)             # 將 LLR 解成 bit，輸出 hard decision

    @tf.function # Graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        no = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=CODERATE, resource_grid=RESOURCE_GRID)

        # Transmitter
        bits = self.binary_source([batch_size, NUM_UT, RESOURCE_GRID.num_streams_per_tx, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)

        # Channel
        y, h_freq = self.channel(x_rg, no)

        # Receiver
        if self.perfect_csi:
            h_hat, err_var = h_freq, 0.
        else:
            h_hat, err_var = self.ls_est (y, no)
        x_hat, no_eff = self.lmmse_equ(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        bits_hat = self.decoder(llr)

        return bits, bits_hat
```

#### `__init__()`: 定義整體通訊模組  
這部分會定義這個class
* 在傳送端: 要產生多少bit、如何做通道編碼、編碼後的bit要用哪種方式mapping (e.g. QAM)等等
* 在接收端: 要如何估計通道、如何Equalizer、如何demapping等等

#### `call()`: 輸入 batch_size, ebno_db 呼叫主程式
這是一個端對端的模擬流程，會從生成位元 → 編碼 → 映射 → OFDM Resource Grid 映射 → 傳送 → 接收 → 等化 → 解調 → 解碼，最後輸出原始位元與解碼後的位元，讓你做錯誤率分析  

#### 如何使用這個 class
1. 定義系統: `model = OFDMSystem(perfect_csi=True)`
2. 呼叫主程式，並輸入參數 `batch_size, ebno_db`，這會自動呼叫 __call__()，完成：bit → 編碼 → QAM → OFDM → 通道 → 等化 → 解調 → 解碼
<img width="1368" height="900" alt="image" src="https://github.com/user-attachments/assets/276b20ad-bc69-42ea-8499-acd61535ea02" />



















