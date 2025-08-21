# Part 1: Getting Started with Sionna


## Communication Systems as Sionna Blocks 

在 Sionna 中，為了更方便建立完整的通訊系統模，通常會將一整套收發流程包裝成一個「class」  
- 如:位元產生 → 映射 → 通道 → 解調


之後只要輸入變數，就能根據該class內的流程產生輸出，換句話說，我們可以事先定義好模型要用的模組、通道、調變器等元件  
- 在 __init__() 中初始化
- 在 __call__() 中呼叫 

### 範例如下:  
```python
class UncodedSystemAWGN(sionna.phy.Block):
    def __init__(self, num_bits_per_symbol, block_length):
        super().__init__()

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sionna.phy.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.awgn_channel = sionna.phy.channel.AWGN()

    def call(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sionna.phy.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y,no)
        return bits, llr
```


***

#### 定義一個新的Class，名為UncodedSystemAWGN:**  
```python
class UncodedSystemAWGN(sionna.phy.Block):
```
這個 `UncodedSystemAWGN` 首先會呼叫 `sionna.phy.Block` ，代表 `UncodedSystemAWGN` 繼承了 `sionna.phy.Block` 的功能  
所以這個 `class` 會變成一個 Sionna Block，能被其他 Block 串接與訓練  

##### def __init__()
```python
def __init__(self, num_bits_per_symbol, block_length):
```
`__init__()`  
是 Python class 中的固定用法，可以想像成你在創建tranceiver的時候，要先設定兩個東西：  
* num_bits_per_symbol：一個 symbol 裡面包含幾個 bit
* block_length：一次傳送幾個 bit
 
`super().__init__()`  
用來把 `sionna.phy.Block` 內的參數初始化  

`self.變數名稱 = 變數 ` :為了在 class 裡面記住自己有哪些變數跟功能，換句話說在這邊會自己定義這個 block 裡會用到哪些元件（像是 Mapper, Demapper）
```python
self.num_bits_per_symbol = num_bits_per_symbol  # 我要用幾個 bit 做成一個 symbol
self.block_length = block_length # 每一筆訊息有多長
self.constellation = sionna.phy.mapping.Constellation("qam", self.num_bits_per_symbol) # 調變的方式:QAM, 每幾個bit組合成一個symbol
self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation) # 使用前一行的constellation，創造一個mapper，這個mapper可以把bit vector直接轉換成星座圖
self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation) # demapper的形式
self.binary_source = sionna.phy.mapping.BinarySource() # 定義一個binary source，來創建 0,1 位元訊號
self.awgn_channel = sionna.phy.channel.AWGN() # 創造一個 AWGN 通道
```
***
##### def call()
```python
def call(self,batch_size, ebn0_db)
```  
在 `sionna.phy.Block` 中， `call()` 是預留給你用來定義這個block「執行時做什麼」的地方  
* `self`: 為了在 class 裡面記住自己有哪些變數跟功能，因此會使用 `self.變數名稱 = 變數 ` 
* `batch_size`: 一共有幾筆bit vector
* `ebn0_db`: 這次模擬使用的Eb/N0

`sionna.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=1.0)`  
會根據括號內的參數，自動計算出相對應的N0  
* ebno_db: 輸入的Eb/N0，單位為dB
* num_bits_per_symbol: 單位符號的bit，會根據def __init__()內的設定而改變
* coderate: 編碼率，實際傳送的bit/經過通道編碼後的bit。若為1代表沒有編碼  

`bits = self.binary_source([batch_size, self.block_length])`  
用`binary_source`隨機產生一組shape為[batch_size, self.block_length]的bit data  

`x=self.mapper(bits)`  
根據__init__()設定的modulation，把bit data mapping到星座圖上

`y=self.awgn_channel(x,N0)`  
把剛剛mapping到星座圖上星座點，送進AWGN通道內

`llr = self.demapper(y,no)`  
針對接收符號y，透過 LLR(log-likelihood ratio)， 把symbol轉回成0,1的bit data  

`return bits, llr`  
* bits: 傳送端產生的原始binary_source
* llr: 接收端針對y去demapper後，針對產生的0,1的估計信心值
***
### 如何使用class UncodedSystemAWGN?
現在我們已經定義好一個class了，現在要來使用它
#### Step1: 建立一個模型，並初始化它
```python
model = UncodedSystemAWGN(num_bits_per_symbol=2, block_length=1024)
```
這時:  
* `__init__()`就會被呼叫
* 會根據我們輸入的`num_bits_per_symbol`和`block_length`初始化這個模型  
#### Step2: 呼叫這個模型來模擬一筆資料傳輸
```python
bits, llr = model(batch_size=2000, ebno_db=5.0)
```
這時:  
* `call()`函數會被執行
* 會根據我們指定的`batch size` 和 `Eb/N₀`
* 產生隨機的bit vector -> symbol(星座點) -> AWGN -> demapping後計算LLR -> 回傳 `(bits,llr)`  

#### 補充說明:  
* num_bits_per_symbol、block_length → 在你**建立模型時**自己指定
* batch_size、ebno_db → 在你**呼叫模型時**給定的輸入參數
***
### 針對uncoded系統在AWGN通道下的BER  
```python
model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=1024)  

EBN0_DB_MIN = -3.0 # Minimum value of Eb/N0 [dB] for simulations
EBN0_DB_MAX = 5.0 # Maximum value of Eb/N0 [dB] for simulations
BATCH_SIZE = 2000 # How many examples are processed by Sionna in parallel

ber_plots = sionna.phy.utils.PlotBER("AWGN")
ber_plots.simulate(model_uncoded_awgn,
                  ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20),
                  batch_size=BATCH_SIZE,
                  num_target_block_errors=100, # simulate until 100 block errors occured
                  legend="Uncoded",
                  soft_estimates=True,
                  max_mc_iter=100, # run 100 Monte-Carlo simulations (each with batch_size samples)
                  show_fig=True);
```
* `ber_plots = sionna.phy.utils.PlotBER("AWGN")`: 建立一個BER模擬器以及繪圖器
* `ber_plots.simulate()`: 使用剛剛定義的繪圖器進行模擬
* `model_uncoded_awgn`: 建立一個模型，即上面建立的 UncodedSystemAWGN
* `ebno_dbs=np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)`: 以 numpy 產生從 -3 到 5dB 的 20 點 Eb/N₀
* `batch_size=BATCH_SIZE`: 一共有幾筆block (一個block即為一個bit vector)
* `num_target_block_errors`: 每個symbol至少有一個bit error
* `legend="Uncoded`: 曲線文字
* `soft_estimates=True`: 用 LLR 進行估計
* `max_mc_iter=100`: 最多進行 100 輪蒙地卡羅模擬
* `show_fig=True` :是否畫出 BER 曲線圖，True
接著，Sionna會印出每個 Eb/N₀ 的 BER 與 BLER的表格，同時產生 AWGN & Eb/N0 圖表
<img width="657" height="425" alt="image" src="https://github.com/user-attachments/assets/51d3208e-41d3-4bf7-936c-da51743a240d" />

***

## Forward Error Correction (FEC)
接下來我們會在收發器中新增通道編碼，以增強其對傳輸錯誤的robust  
為此，我們將使用符合 5G 標準的**低密度奇偶校驗 (LDPC) 碼**和**極化碼**  

```python
k = 12
n = 20
encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(k, n)
decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)
```
* `k`: 原始訊號的bit數
* `n`: 編碼後的bit數
* `encoder=sionna.phy.fec.ldpc.LDPC5GEncoder(k,n)`: 對應到LDCP編碼器，針對 k bit 產生 n bit
* `decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(encoder,hard_out=True)`: 上方encoder對應的解碼器，這裡設定 `hard_out=True`表示解碼後輸出為(0,1)，而非soft value (llr)

### 簡單做個範例
```python
BATCH_SIZE = 1 
u = binary_source([BATCH_SIZE, k])
c = encoder(u)

print("Input bits are: \n", u.numpy())
print("Encoded bits are: \n", c.numpy())
```
Input bits are:  
 [[0. 0. 0. 0. 1. 0. 0. 1. 1. 1. 1. 1.]]  
Encoded bits are:  
 [[1. 0. 0. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1. 0.]]  
 * 將原始12-bit經由LDCP編碼後，得到20bit的codeword

***

### 在多用戶、多基地台、多樣本(batch)的架構下，使用 5G LDCP
Sionna也支援多維度的tensor，意思是我們可以同時處理
* 多個用戶
* 多根天線
* 多個樣本

```python
BATCH_SIZE = 10
num_basestations = 4
num_users = 5 
n = 1000 # codeword length per transmitted codeword
coderate = 0.5 # coderate
k = int(coderate * n) # number of info bits per codeword

encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(k, n)
decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(encoder,
                                    hard_out=True, # 輸出為0,1，否則為LLR
                                    return_infobits=True, # 是否只輸出資訊位元，否則 also return (decoded) parity bits
                                    num_iter=20, # 解碼演算法迭代次數
                                    cn_update="boxplus-phi") # 檢查節點更新方式

# 產生輸入資料並編碼
u = binary_source([BATCH_SIZE, num_basestations, num_users, k])
c = encoder(u)

print("Shape of u: ", u.shape)
print("Shape of c: ", c.shape)
print("Total number of processed bits: ", np.prod(c.shape))
```
Shape of u:  (10, 4, 5, 500)  
Shape of c:  (10, 4, 5, 1000)  
Total number of processed bits:  200000  

### 將 LDPC code 替換為 Polar code
```python
k = 64
n = 128
encoder = sionna.phy.fec.polar.Polar5GEncoder(k, n)
decoder = sionna.phy.fec.polar.Polar5GDecoder(encoder,
                                      dec_type="SCL") # you can also use "SCL"
```
這樣的寫法會自動處理以下步驟:    
* Rate matching: 根據你指定的(n,k)，自動調整實際傳輸位元
* CRC concatenation: 自動在資訊位元後面加上**CRC bit**，讓解碼時能除錯
* `dec_type="SLC`: 內建選擇解碼器類型，此次範例為 `SCL` 代表使用 Successive Cancellation List 解碼器
這就是標準 5G Polar code chain 的作法，整體流程是符合 3GPP 標準的，你不需要再手動控制 CRC 或 rate-matching

***

## Eager vs Graph Mode — 動態執行 vs 圖模式加速執行
目前為止，這個教學中的程式碼都在 Eager Mode（即時模式） 下執行
* Eager mode 是 TensorFlow 的一種執行模式，讓你像使用 NumPy 那樣，逐行執行每一個 TensorFlow 的操作
* 好處是便於除錯與開發，因為你可以即時觀察每一步的輸出結果
* 這對初學者來說非常直覺，但在運行大型模擬（如 Sionna 通訊系統）時，效率就不如 Graph mode。  

若要完整發揮 Sionna 在 通訊系統模擬 中的計算效能，我們需要啟用 Graph mode（圖模式）
* Graph mode 是 TensorFlow 的另一種執行方式
* 會先把你的 Python 函式「編譯」成靜態的計算圖（computational graph）
* 然後在底層以最佳化方式執行，通常有顯著的加速效果，尤其是大量批次資料（batch）處理

### 開啟方式
```python
@tf.function()
```
舉例說明  
```python
@tf.function()  
def run_graph(batch_size, ebno_db):
    print(f"Tracing run_graph for values batch_size={batch_size} and ebno_db={ebno_db}.")
    return model_coded_awgn(batch_size, ebno_db)
```
1. `@tf.function()`: 開啟 Graph mode。這段函式的所有TensorFlow操作都會變成靜態圖的一部分
2. `def run_graph(batch_size, ebno_db)`: 定義一個叫做 run_graph 的函式，輸入為兩個參數
3. `print()`: 因為現在是 Graph mode ，**所以這行只會在 trace階段印出一次，也就是說，只要輸入類型或形狀沒改變，就不會再次印出這行**
4. `return model_coded_awgn()`: 回傳函式的結果

反例: 即使你有 @tf.function，但若傳入的是 Python 數值而非 Tensor，還是會每次 retrace  
```python
batch_size = 10
ebno_db = 1.5
run_graph(batch_size, ebno_db)
```

#### 第一次呼叫 run_graph() 時  
print() 這一行會被執行一次，因為 TensorFlow 會對這個函式進行 trace，並把它轉換成一個靜態的計算圖  
什麼是 Trace?  
* TensorFlow 為了將 Python 函式變成Graph，第一次執行時會根據當下的輸入資料型別與形狀，建構對應的計算圖
* 這個過程就叫做tracing
* 當 trace 完成後，未來如果你再次用相同資料型別與形狀呼叫它，TensorFlow 就會重用這個靜態圖，而不會重複執行 print()這個 python code
* **所以即便更改了輸入的值，只要資料型別與形狀的話，就不用重新Graph一次**
* **未來只需利用先前建立好的 Graph來進行計算即可，因為已經編譯好了**

#### 第二次呼叫 run_graph() 時  
由於輸入的形狀與型別沒有改變，因此會使用第一次 trace 過後所產生的 Graph  
* 這正是 Graph mode 的效能優勢來源：只需要建立一次圖，就能在之後重複使用。
* 也就是說：你的 Python print() 不會再執行，因為它只在 trace 階段執行一次。



