# Part 5: Basic MIMO Simulations
* 在本篇中，您將學習如何設定 **flat fading channel**上的 MIMO 傳輸模擬  
* 以下是包含所有必要組件的系統模型示意圖
<img width="650" height="400" alt="image" src="https://github.com/user-attachments/assets/308105ac-3530-49a1-99f9-ffed1601c0f8" />

在本章內容，您會學習如何: 
* 使用 `FastFadingChannel` class
* 使用 spatial antenna correlation
* 在 CSI 的情況下實現 LMMSE 檢測
* 錯誤率模擬

***

## 教學章節流程
* GPU Configuration and Imports
* Simple uncoded transmission: Adding spatial correlation
* Extension to channel coding: BER simulations using a Sionna Block

*** 

## GPU Configuration and Imports

```python
import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
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

# Set random seed for reproducability
sionna.phy.config.seed = 42
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sionna.phy import Block
from sionna.phy.utils import ebnodb2no, compute_ser, compute_ber, PlotBER
from sionna.phy.channel import FlatFadingChannel, KroneckerModel
from sionna.phy.channel.utils import exp_corr_mat
from sionna.phy.mimo import lmmse_equalizer
from sionna.phy.mapping import SymbolDemapper, Mapper, Demapper, BinarySource, QAMSource
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
```

***

## Simple uncoded transmission
流程: **QAM 發送** → Rayleigh 通道 + AWGN → 接收 → LMMSE 等化 → 還原星座點 → 計算 SER
* 我們將考慮從具有 num_tx_ant 個天線的發射機到具有 num_rx_ant 個天線的接收機的點對點傳輸  
* 發射機不採用預編碼，並從每個天線發送獨立的資料流  
### 產生一批 16-QAM 的發射向量

```python
num_tx_ant = 4                                # TX 天線
num_rx_ant = 16                               # RX 天線 
num_bits_per_symbol = 4                       # 16-QAM
batch_size = 1024                             # 1024 組傳送訊號
qam_source = QAMSource(num_bits_per_symbol)   # (1024, 4)
x = qam_source([batch_size, num_tx_ant])      # 輸出為 (1024, 4)
print(x.shape)                                
```
* `QAMSource(4)`: Sionna中的模組，建立 QAM symbol 產生器 (例如 16QAM 就是實部 ±1, ±3 和虛部 ±1j, ±3j 組合)
* qam_source([1024,4]) : 產生 1024 組 data， 每組有4個 QAM symbol
  (可以想成有1024個時間點，並且由4根TX_ant發送4個symbol)

### 建立 flat fading channel + AWGN + 接收
流程: QAM 發送 → **Rayleigh 通道 + AWGN → 接收** → LMMSE 等化 → 還原星座點 → 計算 SER 
用來模擬在 i.i.d. Rayleigh fading channel 的傳輸  

```python
channel = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=True) 
no = 0.2 # Noise variance of the channel

# y and h are the channel output and channel realizations, respectively.
y, h = channel(x, no)
print(y.shape) 
print(h.shape)
```
* x.shape: (1024,4)   
* y.shape: (1024,16)
* h.shape: (1024,16,4)

### LMMSE Equalizer
流程: QAM 發送 → Rayleigh 通道 + AWGN → 接收 → **LMMSE 等化 → 還原星座點** → 計算 SER  



```python
s = tf.cast(no*tf.eye(num_rx_ant, num_rx_ant), y.dtype)  # 告訴equalizer雜訊的共變數矩陣
x_hat, no_eff = lmmse_equalizer(y, h, s)                 # 執行 LMMSE equalizer

plt.axes().set_aspect(1.0)
plt.scatter(np.real(x_hat), np.imag(x_hat));
plt.scatter(np.real(x), np.imag(x));
```

因為有 perfect CSI， 我們現在可以實作一個 LMMSE equalizer 來計算 soft symbol  
假設你接收到一個點在「(2.7 + j0.9)」附近
* 如果是 Hard symbol：你會直接判成 (3 + j1)，然後對應 bit pattern
* 如果是 Soft symbol：你會保留「2.7 + j0.9」這個值，讓解碼器根據這個距離來決定 bit 的機率
<img width="436" height="413" alt="image" src="https://github.com/user-attachments/assets/c6199341-0647-4210-bbd2-1f47501aab23" />

如上圖所示  
* soft symbol `x_hat`分散在 16-QAM 星座點周圍
* Equalizer 的輸出 no_eff ，為每個 soft symbol 提供雜訊方差的估計值

### 根據星座圖上的 soft symbol 做 hard decisions 並計算SER
流程: QAM 發送 → Rayleigh 通道 + AWGN → 接收 → LMMSE 等化 → 還原星座點 → **計算 SER**
* 我們目前只有 LMMSE 等化器輸出的 soft symbols（可能偏離 QAM 星座點），不能直接當成判斷結果
* 所以我們需要將這些值歸類到最接近的 QAM 星座點，這個動作就叫 **hard decision**
* 最後，將這些**hard decision**後的符號與實際發送的符號做比對，就可以得到符號錯誤率（SER）

```python
symbol_demapper = SymbolDemapper("qam", num_bits_per_symbol, hard_out=True)

# Get symbol indices for the transmitted symbols
x_ind = symbol_demapper(x, no)

# Get symbol indices for the received soft-symbols
x_ind_hat = symbol_demapper(x_hat, no)

compute_ser(x_ind, x_ind_hat)
```

***

## Extension to channel coding
* MIMO 系統中的天線不總是獨立的，實際上因為天線間距不夠遠，信號會互相干擾，產生「空間相關性」
* 這種情況會讓通道矩陣 H 裡的元素之間產生相關（correlation），不再獨立
*Sionna 提供 `SpatialCorrelation` 模組，透過 `Kronecker correlation model` 與指數相關矩陣 `exp_corr_mat`，模擬實際通道

```python
# 建立發送與接收端的相關性矩陣
r_tx = exp_corr_mat(0.4, num_tx_ant)
r_rx = exp_corr_mat(0.9, num_rx_ant)

# 使用 Kronecker model 加入空間相關性到通道模型
channel.spatial_corr = KroneckerModel(r_tx, r_rx)
```

* `exp_corr_mat(ρ, N)`: 產生一個 N × N 的 correlation matrix

### 接下來，我們可以透過建立大量通道實作來驗證通道模型是否應用了所需的空間相關性
```python
h = channel.generate(1000000)

# Compute empirical covariance matrices
r_tx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_a=True), 0)/num_rx_ant
r_rx_hat = tf.reduce_mean(tf.matmul(h, h, adjoint_b=True), 0)/num_tx_ant

# Test that the empirical results match the theory
assert(np.allclose(r_tx, r_tx_hat, atol=1e-2))
assert(np.allclose(r_rx, r_rx_hat, atol=1e-2))
```
* `h = channel.generate(1000000)`: 產生 100 萬筆通道樣本，我們要用來觀察實際產生的通道矩陣 h 是否真的符合我們設定的空間相關性

### 將transmit symbol透過有spatial correlation的通道進行傳輸，並計算SER
```python
y, h = channel(x, no)
x_hat, no_eff = lmmse_equalizer(y, h, s)  # 使用 LMMSE 等化器根據接收訊號 y 和通道 h，估計傳送符號 x_hat
x_ind_hat = symbol_demapper(x_hat, no)    # 將 x_hat做hard decision，映射回 16-QAM 星座點
compute_ser(x_ind, x_ind_hat)             # 將原始符號 x_ind 與預測符號 x_ind_hat 進行比較，計算 SER
```
輸出: 大約有 12.06% 的符號錯誤率  
`<tf.Tensor: shape=(), dtype=float64, numpy=0.12060546875>`
這個結果清楚地顯示了在此設定中，空間相關性對通訊效能的負面影響  

*** 

## Extension to channel coding
* 到目前為止，我們已經模擬了**非編碼符號**的傳輸。  
* 接下來只需添加幾行程式碼，我們就可以擴展我們的工作，以模擬**編碼的BER**

#### Step 1. 使用 LDPC block code 編碼
```python
n = 1024 # codeword length
k = 512  # number of information bits per codeword
coderate = k/n # coderate
batch_size = 32
```
#### Step 2. 建立元件: Bit source, encoder/decoder 
```python
binary_source = BinarySource()
encoder = LDPC5GEncoder(k, n)
decoder = LDPC5GDecoder(encoder, hard_out=True)
mapper = Mapper("qam", num_bits_per_symbol)
demapper = Demapper("app", "qam", num_bits_per_symbol)
```
#### Step 3. 透過編碼位元映射，來產生隨機 QAM 符號
```python
b = binary_source([batch_size, num_tx_ant, k])   # 產生原始 bit
c = encoder(b)                                   # 經過 LDPC 編碼，長度從k → n
x = mapper(c)                                    # 經過 QAM 映射，從 bits → complex symbols  
x_ind = symbol_demapper(x, no)                   # 存下發送端 symbol index ，用來計算後續的 SER
shape = tf.shape(x)                              # 保留原始傳送訊號的shape，用於 step 5 解碼
x = tf.reshape(x, [-1, num_tx_ant])              # -1: 這一維的大小由 TensorFlow 根據其他維度自動計算出來   
print(x.shape)                                   # (8192, 4)   
```
* `b.shape`: [32,4,512] (產生了 32 組資料，每組有 4 根天線，每根天線負責發送 512 個 bit（未編碼）)
* `c.shape`: [32,4,1024] (每根天線原本 512 bit，經過 LDPC 編碼變成 1024 bit)
* `x.shape`: [32,4,256] (因為每個 QAM symbol 代表 4 bit（16-QAM），所以1024/4=256)
* 代表 x 有 32 組樣本；每組 4 根天線、每根天線發送 256 個 QAM symbols
* `x = tf.reshape(x, [-1, num_tx_ant])`: 代表把原本三維的 x reshape 成二維

***

#### Step 4. 將發射端編碼後的symbol傳送到通道中
```python
y, h = channel(x, no)
x_hat, no_eff = lmmse_equalizer(y, h, s)
```

#### Step 5. 將接收端的符號映射到 LLR，並解碼
在這之前，需要把之前 reshape 的訊號還原成原始的 shape ，即(8192,4) -> (32,4,1024)
```python
x_hat = tf.reshape(x_hat, shape)
no_eff = tf.reshape(no_eff, shape)
```

demapper 根據 `x_hat` 和 `no_eff` ，輸出每個 bit 的 Log-Likelihood Ratio (LLR)
將 llr 輸入進 LDPC decoder，得到預測的位元序列 b_hat
```python
llr = demapper(x_hat, no_eff)
b_hat = decoder(llr)
```

```python
x_ind_hat = symbol_demapper(x_hat, no)                           # 把 soft-symbol demap到最接近的星座點
ber = compute_ber(b, b_hat).numpy()                              # 錯誤率計算
print("Uncoded SER : {}".format(compute_ser(x_ind, x_ind_hat)))
print("Coded BER : {}".format(compute_ber(b, b_hat)))
```
輸出:  
```python
Uncoded SER : 0.120452880859375
Coded BER : 0.0
```
* 即使符號錯誤率高達 12%，但經過 LDPC 解碼後位元錯誤率為 0%
* 這顯示通道編碼能大幅提升可靠度

## BER simulations using a Sionna Block
最後，把目前為止所做的所有工作封裝到一個 **Sionna Block** 中，以便於進行 BER 模擬和系統參數比較

```python
class Model(Block):
    def __init__(self, spatial_corr=None):
        super().__init__()
        self.n = 1024
        self.k = 512
        self.coderate = self.k/self.n
        self.num_bits_per_symbol = 4
        self.num_tx_ant = 4
        self.num_rx_ant = 16
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
        self.channel = FlatFadingChannel(self.num_tx_ant,
                                         self.num_rx_ant,
                                         spatial_corr=spatial_corr,
                                         add_awgn=True,
                                         return_channel=True)

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        b = self.binary_source([batch_size, self.num_tx_ant, self.k])
        c = self.encoder(b)

        x = self.mapper(c)
        shape = tf.shape(x)
        x = tf.reshape(x, [-1, self.num_tx_ant])

        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)
        no *= np.sqrt(self.num_rx_ant)

        y, h = self.channel(x, no)
        s = tf.complex(no*tf.eye(self.num_rx_ant, self.num_rx_ant), 0.0)

        x_hat, no_eff = lmmse_equalizer(y, h, s)

        x_hat = tf.reshape(x_hat, shape)
        no_eff = tf.reshape(no_eff, shape)

        llr = self.demapper(x_hat, no_eff)
        b_hat = self.decoder(llr)

        return b,  b_hat
```
<img width="658" height="421" alt="image" src="https://github.com/user-attachments/assets/22a44c9d-94ca-4297-9ddc-2f2a5ea1a308" />
