<img width="1796" height="89" alt="image" src="https://github.com/user-attachments/assets/349737b7-5dce-4f5d-88d8-013bb17ad98b" /># Part 0: “Hellow world!”
這章節用非常簡單的例子，模擬 16-QAM 傳送到 AWGN 的樣子  
此章節的內容在後面幾章都有更詳細的介紹，因此可以跳過，直接從 part 1 開始  

## Import
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

# IPython "magic function" for inline plots
%matplotlib inline
import matplotlib.pyplot as plt
```

***

## Step 1. 首先藉由`BinarySource`來產生一批隨機的位元向量
**產生bit vector** -> 建立星座圖 -> 將 bit vector mapping 到星座圖上  
```python
batch_size = 1000             # 產生1000筆資料
num_bits_per_symbol = 4       # 每筆資料為4bit vector (16-QAM) 
binary_source = sionna.phy.mapping.BinarySource()      
b = binary_source([batch_size, num_bits_per_symbol]) 
```
* `sionna.phy.mapping.BinarySource()`: 專門用來產生0 與 1 的資料，並取名為 binary_source
* `binary_source([1000,4])`: 產生1000筆長度為4的vector
### 輸出

```python
<tf.Tensor: shape=(1000, 4), dtype=float32, numpy=
array([[0., 0., 0., 0.],
       [1., 1., 1., 1.],
       [0., 1., 1., 1.],
       ...,
       [1., 1., 1., 1.],
       [1., 1., 1., 0.],
       [0., 1., 1., 0.]], dtype=float32)>
```
## Step 2. 建立一個星座圖
產生bit vector -> **建立星座圖** -> 將 bit vector mapping 到星座圖上  
```python
constellation = sionna.phy.mapping.Constellation("qam", num_bits_per_symbol)
constellation.show();
```
* `sionna.phy.mapping.Constellation("調變方式",bit/symbol)`: 代表星座圖使用 QAM ，並且每個 symbol 由 4-bit 組成，因此為 16-QAM
<img width="621" height="624" alt="image" src="https://github.com/user-attachments/assets/cf9b06a6-aa9f-476c-ba55-16e7a1b7f8f7" />

## Step 3. 根據先前建立的星座圖，
產生bit vector -> 建立星座圖 -> **將 bit vector mapping 到星座圖上**  
```python
mapper = sionna.phy.mapping.Mapper(constellation=constellation) 
x = mapper(b)
x[:10]
```
* `sionna.phy.mapping.Mapper(constellation=constellation)`: 根據先前定義的星座圖，建立一個對應的mapper
* `x = mapper(b)`: 把`BinarySource``b`，透過上述的mapper映射到對應的星座圖上

### 輸出
```python
<tf.Tensor: shape=(10, 1), dtype=complex64, numpy=
array([[ 0.31622776+0.31622776j],
       [-0.94868326-0.94868326j],
       [ 0.94868326-0.94868326j],
       [-0.31622776+0.94868326j],
       [ 0.94868326-0.31622776j],
       [ 0.94868326+0.31622776j],
       [ 0.31622776-0.31622776j],
       [-0.31622776-0.31622776j],
       [-0.31622776-0.94868326j],
       [-0.31622776-0.31622776j]], dtype=complex64)>
```

## Step 4. 加入AWGN
```python
awgn = sionna.phy.channel.AWGN()
ebno_db = 15 
no = sionna.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1)
y = awgn(x, no)

# Visualize the received signal
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
plt.scatter(np.real(y), np.imag(y));
ax.set_aspect("equal", adjustable="box")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True, which="both", axis="both")
plt.title("Received Symbols");
```
<img width="612" height="624" alt="image" src="https://github.com/user-attachments/assets/6eb7153e-502a-4c1c-9ae9-608aa24bbde6" />

