# Sionna-PHY
## Introduction to Sionna PHY
待編輯

***
## Part 0: “Hellow world!”
待編輯
 
***
## Part 1: Getting Started with Sionna


### Communication Systems as Sionna Blocks 

在 Sionna 中，為了更方便建立完整的通訊系統模，通常會將一整套收發流程包裝成一個「class」  
- 如:位元產生 → 映射 → 通道 → 解調


之後只要輸入變數，就能根據該class內的流程產生輸出，換句話說，我們可以事先定義好模型要用的模組、通道、調變器等元件  
- 在 __init__() 中初始化
- 在 __call__() 中實作前向推論（forward pass)  


範例如下:  
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

**第一步：你定義一個新的系統 UncodedSystemAWGN:**  
```python
class UncodedSystemAWGN(sionna.phy.Block):
```
這個 `UncodedSystemAWGN` 首先會呼叫 `sionna.phy.Block` ，代表 `UncodedSystemAWGN` 繼承了 `sionna.phy.Block` 的功能  
所以這個 `class` 會變成一個 Sionna Block，能被其他 Block 串接與訓練  

**第二步：呼叫 super().__init__():**  
```python
super().__init__():
```
這行就是把 `sionna.phy.Block` 的東西初始化  
**第三步：自己定義這個 block 裡會用到哪些元件（像是 Mapper, Demapper）:**  
* 為了在 class 裡面記住自己有哪些變數跟功能，因此會使用 `self.變數名稱 = 變數 `  
```python
self.num_bits_per_symbol = num_bits_per_symbol  # 我要用幾個 bit 做成一個 symbol
self.block_length = block_length # 每一筆訊息有多長
self.constellation = sionna.phy.mapping.Constellation("qam", self.num_bits_per_symbol) # 調變的方式:QAM, 每幾個bit組合成一個symbol
self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation) # 使用前一行的constellation，創造一個mapper，這個mapper可以把bit vector直接轉換成星座圖
self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation) # demapper的形式
self.binary_source = sionna.phy.mapping.BinarySource() # 定義一個binary source，來創建 0,1 位元訊號
self.awgn_channel = sionna.phy.channel.AWGN() # 創造一個 AWGN 通道
```




  

#### Sionna block code



