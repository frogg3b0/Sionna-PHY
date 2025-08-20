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

