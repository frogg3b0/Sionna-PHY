# Link-level simulations with Sionna RT
* 在本筆記本中，您將使用 **ray-traced channels** 進行鏈路級模擬
* 而不是隨機通道模型 （如 i.i.d. Rayleigh 或 CDL）

***

## Background Information
### Ray Tracing 是甚麼?
* 模擬「無線電波從發射端到接收端在環境中的**路徑**」
* 這當中，會考慮環境中的建築物、遮蔽、反射等現象，來產生「更物理實際」的通道模型
* 相較於理想化的 Rayleigh 或 CDL 通道，ray tracing 模型可以模擬真實環境條件下的通道特性

### Sionna RT
* 是 Sionna 的一個模組，專門處理 ray tracing（建構於 Mitsuba 3）

### 本章節模擬系統
* 我們使用 [5G NR PUSCH 教程筆記](https://nvlabs.github.io/sionna/phy/tutorials/5G_NR_PUSCH.html)中的 5G NR PUSCH 發射機和接收機
* 評估 MU‑MIMO 5G NR uplink 的 BER 表現
* 由於頻道互易性，之後可以反轉 Ray tracing channel 的方向。上行/下行不會改變模擬路徑

***

## Import 
```python
import os # Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import or install Sionna
try:
    import sionna.phy
    import sionna.rt
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

# For link-level simulations
from sionna.phy.channel import OFDMChannel, CIRDataset
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.phy.utils import ebnodb2no, PlotBER
from sionna.phy.ofdm import KBestDetector, LinearDetector
from sionna.phy.mimo import StreamManagement

# Import Sionna RT components
from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                      PathSolver, RadioMapSolver

no_preview = True # Toggle to False to use the preview widget
                  # instead of rendering for scene visualization
```

***

## Setting up the Ray Tracer
### Step 1. 通訊系統參數定義
```python
# System parameters
subcarrier_spacing = 30e3       # 子載波間距，設定為 30 kHz。
num_time_steps = 14             # 每個 slot 有 14 個 OFDM 符號

num_tx = 4                      # 系統中有 4 個 UE
num_rx = 1                      # 模擬中只考慮 1 個接收端（Base Station）
num_tx_ant =  4                 # 每個用戶設備擁有 4 根天線，對應 MIMO uplink 傳輸
num_rx_ant = 16                 # 接收端（BS）配置了 16 根天線

# batch_size for CIR generation
batch_size_cir = 1000           # 將來在產生 CIR 資料時，一次產生的樣本批量大小 (e.g. 1000 個不同 UE 位置)
```
### Step 2. 無線電波傳播環境設定
* 我們接下來要設定「無線電波傳播環境」（即建立場景、基地台、天線、傳播模型等）
* 從載入一個場景開始，接著新增一個發射器，此發射器將扮演基地台的角色
* 之後我們會利用「通道互易性」來模擬上行方向（UE→BS）的通道
細節可參閱 Sionna RT 內容
#### Step 2.1. 載入整合場景並定義天線參數
```python
scene = load_scene(sionna.rt.scene.munich)

# Transmitter (=basestation) has an antenna pattern from 3GPP 38.901
scene.tx_array = PlanarArray(num_rows=1,             # 垂直方向只有一排，表示橫向展開的 ULA  
                             num_cols=num_rx_ant//2, # 幾何上的單元數量
                             vertical_spacing=0.5,   # 天線間格為半波長
                             horizontal_spacing=0.5,
                             pattern="tr38901",      # 套用 3GPP TR 38.901 pattern
                             polarization="cross")   # 每個單元有兩個交叉極化天線（e.g. ±45°）
```
* `load_scene(內建場景)`: 是 Sionna RT 提供的函式，用來載入內建的 3D 無線通訊場景，包含建物形狀、材質（反射係數)
* `num_cols=num_rx_ant//2`: 設定的是「幾何上的單元數量」，而天線總數(16) = 幾何單元數(8) × 極化數(2)
* `vertical_spacing=0.5`: Sionna RT 會自動把這解釋為「0.5 倍波長」的間距，而不是 0.5 公尺或其他單位

#### Step 2.2. 在場景內建立一個 TX 物件
```python
# Create transmitter
tx = Transmitter(name="tx",
                 position=[8.5,21,27],  # TX 的三圍座標位置
                 look_at=[45,90,1.5],   # TX 的主波束、主要射線、最大天線增益導向的方向
                 display_radius=3.)     # 可視化時，此 TX 所用的球體半徑（僅影響視覺呈現，與實際電磁建模無關）。
scene.add(tx)                           # 將上面建立的 TX 加入到目前的 scene 場景中

# Create new camera 這部分不影響物理性質，單純是為了在 notebook 或可視化工具中可以看到整個場景的樣貌
bird_cam = Camera(position=[0,80,500], orientation=np.array([0,np.pi/2,-np.pi/2]))  # 從某個位置[0,80,500]，用某個視角(鳥瞰)去觀察該場景
```

### Step 3. 針對這個 TX，去計算對應的 Radio map
```python
max_depth = 5                           # 設定每條 ray 最多可以經過 5 次「反射、繞射或穿透」，並非 path 數量
rm_solver = RadioMapSolver()            # 建立一個 RadioMapSolver 物件，它是 Sionna RT 中用來「計算場域中接收強度分布」的工具
rm = rm_solver(scene,                   # 呼叫 RadioMapSolver 物件，傳入參數來計算 radio map。回傳結果的物件為rm
               max_depth=5,
               cell_size=(1., 1.),      # 把整個地面區域劃成 1 公尺 × 1 公尺 的小方格，對每個格子進行取樣與統計
               samples_per_tx=10**7)    # 每個 TX 的 path 總數
```

### Step 4. 將上述場景可視化
```python
if no_preview:
    # 靜態渲染圖像
    scene.render(camera=bird_cam,  # 使用這個視角，靜態觀察場景
                 radio_map=rm,     # 要渲染的物件 = rm
                 rm_vmin=-110,     # 可視化畫面中，最小功率值
                 clip_at=12.);     # 將建物、地面以外高度大於 12 公尺的區域切除，方便觀察地面功率分布。
else:
    # 互動式視覺預覽
    scene.preview(radio_map=rm,
                  rm_vmin=-110,
                  clip_at=12.); 
```
 `if no_preview `: 
 * `True`: 使用**靜態渲染圖像**
 * `False`: 使用**互動式視覺預覽（widget）**，可滑動、旋轉視角
<img width="766" height="590" alt="image" src="https://github.com/user-attachments/assets/7fc1c229-41bc-42ed-bba1-4f5e99df54ed" />

### Step 5. 從 Radio map 篩選合理的UE位置候選點
```python
min_gain_db = -130
max_gain_db = 0 
min_dist = 5 
max_dist = 400 

# 從 Radio Map 符合條件的「可通訊位置集合」中，隨機抽樣出 `num_pos=batch_size_cir` 個點作為**UE 實際擺放的位置**
ue_pos, _ = rm.sample_positions(num_pos=batch_size_cir,
                                metric="path_gain",
                                min_val_db=min_gain_db,
                                max_val_db=max_gain_db,
                                min_dist=min_dist,
                                max_dist=max_dist)
```
根據這個Radio map，我們會自動略過:  
* path gain 低於 -130 dB 的 path
* path gain 超過 0 db 的 path
* 距離 RX 超過 400m 的位置
* 距離 RX 小於 5m 的位置
* 所有符合這些條件的離散 points 構成候選集合（可能數千、數萬個）

接著根據上述條件，從 Radio Map 符合條件的「可通訊位置集合」中，隨機抽樣出 `num_pos=batch_size_cir` 個點作為**UE 實際擺放的位置**

### Step 6. 將 UE 位置轉化為場景中的 Receiver 物件
#### Step 6.1. 為所有接收器（UE）統一指定一個天線陣列樣式
```python
scene.rx_array = PlanarArray(num_rows=1,             # 設定天線幾何形狀為 1×2 的平面陣列 
                             num_cols=num_tx_ant//2, # Each receiver is equipped with 4 antennas
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",          # 天線單元為各向同性（isotropic）
                             polarization="cross")   # 每個天線單元使用雙極化（cross-polarized）
```

#### Step 6.2. Receiver 建立與加入場景
```python
for i in range(batch_size_cir):                            # 建立 batch_size_cir 個 UE Receiver
    scene.remove(f"rx-{i}")                                # 每次重跑前清除舊物件   
    rx = Receiver(name=f"rx-{i}",                          # 給每個 Receiver 命名 (e.g. rx-0, rx-1)   
                  position=ue_pos[0][i],                   # 將第 i 個由 Radio Map 抽樣的位置作為 RX 的位置
                  velocity=(3.,3.,0),                      # 指定該 RX（UE）的速度向量
                  display_radius=1.,                       # 僅用於可視化：畫出 RX 為一個半徑 1 公尺的球體
                  color=(1,0,0)                            # 僅用於可視化：將 RX 顯示為紅色 (R=1,G=0,B=0)
                  )
    scene.add(rx)                                          # 把剛剛創建的 rx Receiver 物件加入當前場景中
```

#### Step 6.3. 可視化場景
```python
# And visualize the scene
if no_preview:
    # Render an image
    scene.render(camera=bird_cam,
                 radio_map=rm,
                 rm_vmin=-110,
                 clip_at=12.); # Clip the scene at rendering for visualizing the refracted field
else:
    # Show preview
    scene.preview(radio_map=rm,
                  rm_vmin=-110,
                  clip_at=12.); # Clip the scene at rendering for visualizing the refracted field
```
<img width="766" height="590" alt="image" src="https://github.com/user-attachments/assets/4adfc7d2-6e73-42eb-ad41-f09ba1be3965" />  

* 每個點代表從 **Radio map** 的隨機取樣函數中，提取的接收機位置
* 使在複雜場景下，這也能有效地對隨機通道實作進行批量採樣

*** 

## Creating a CIR Dataset
根據事前建立好的 Radio map，現在我們就可以模擬**來自多個不同位置 UE 對應的 CIR**  
#### Step 0. 先把觀念講清楚，假設我們想提取 `target_num_cirs = 5000` 個 CIR
* 5000 個 CIR = 5000 個不同的 UE 位置他們各自對應的 ray tracing 路徑所構成的 CIR
    * UE 1 自己本身會有多路徑，因此一個 UE 就會產生 1 個 CIR
    * 因此我們只要找到 5000 個不同位置的 UE， 就能產生 5000 組 CIR
* 具體的做法是:
    * 每次產出 1000 個 CIR (找到1000個不同位置的用戶)
    * 總共需要產生5次不同的模擬結果，這部分會透過`seed=idx`逐次遞增，讓我們每一輪的隨機抽樣都會**使用不同的`seed`**
    * 最後就可以產出 5000 組 各自不同的 CIR
* 觀念釐清之後，接下來進入程式範例

#### Step 1. 初始化變數與 PathSolver 建立
```python
target_num_cirs = 5000    # 定義這次模擬要產生多少個不同的 CIR
max_depth = 5             # ray tracing 可容許的最大交互次數（反射/繞射/透射最多 5 次）
min_gain_db = -130       
max_gain_db = 0
min_dist = 10
max_dist = 400 

# 將多次模擬產生的 CIR 結果統一收集在這兩個 list 中，最後才會組合起來
a_list = []              # 複數通道增益
tau_list = []            # 對應的 delay

# 因為不同位置產生的 CIR 含有的 path 數不同 ，為了之後統一 Tensor 尺寸，需要這個變數來進行 padding 對齊
max_num_paths = 0        # 記錄目前所有 batch 中，最多的路徑數量

p_solver = PathSolver()  # 建立一個射線追蹤的 path solver 物件
```

#### Step 2. 產生目標數量的 CIR
```python
num_runs = int(np.ceil(target_num_cirs/batch_size_cir))                          # 計算所需模擬次數
for idx in range(num_runs):                                                       
    print(f"Progress: {idx+1}/{num_runs}", end="\r")

    # 從符合<條件>的集合中，隨機抽樣 UE 的位置
    ue_pos, _ = rm.sample_positions(                                             
                        num_pos=batch_size_cir,                                  # 一次抽樣 batch_size_cir 個 UE 的位置
                        metric="path_gain",                                      # 根據何種 metric 來決定哪些位置有效
                        min_val_db=min_gain_db,                                  # <條件1> path gain 下限
                        max_val_db=max_gain_db,                                  # <條件2> path gain 上限
                        min_dist=min_dist,                                       # <條件3> 和 TX 距離不能太近
                        max_dist=max_dist,                                       # <條件4> 和 TX 距離不能太遠
                        seed=idx)                                                # 每輪使用不同seed，確保抽樣位置不重複

    # 把這一輪抽出來的使用者位置 -> 指派為接收器
    for rx in range(batch_size_cir):
        scene.receivers[f"rx-{rx}"].position = ue_pos[0][rx]

    # 這一步會輸出所有 TX → RX 的有效路徑，並儲存在 paths 物件中
    paths = p_solver(scene, max_depth=max_depth, max_num_paths_per_src=10**7)   

    # 將 ray tracing 結果轉換為 CIR (channel impulse responses)
    a, tau = paths.cir(sampling_frequency=subcarrier_spacing,                   # 設定時間軸上採樣頻率，這裡用 OFDM 的子載波間距 30kHz
                         num_time_steps=14,                                     # 每個 CIR 有 14 個離散取樣點，對應到 OFDM symbol 數量
                         out_type='numpy')                                      # 輸出格式為 numpy
    a_list.append(a)                                                            # batch 得到的 CIR（a, tau）加進列表
    tau_list.append(tau)

    # Update maximum number of paths over all batches of CIRs
    num_paths = a.shape[-2]
    if num_paths > max_num_paths:
        max_num_paths = num_paths
```
* 至此，收集了多次模擬的結果（每輪產生 batch_size_cir 個 CIR），累積在 `a_list` 和 `tau_list` 中  
* 然而，每一筆 CIR 的 path 數量不同（num_paths 不固定），所以如果你直接合併這些 array，會導致形狀不一致、無法拼接
* 因此，接下來需要
1. 將所有 CIR padding 成統一長度: 使用 `np.pad` 使每個 `a_`, `tau_` 的 path 數都擴充到 `max_num_paths`，缺的部分補 0
2. 將所有模擬批次的 CIR 統整為單一 array
3. 交換 Tx 與 Rx 的維度、調整 batch 維度

#### Step 3. 將不同批次模擬結果統一格式，讓其path數量統一
```python
a = []
tau = []
for a_,tau_ in zip(a_list, tau_list):
    num_paths = a_.shape[-2]
    a.append(np.pad(a_, [[0,0],[0,0],[0,0],[0,0],[0,max_num_paths-num_paths],[0,0]], constant_values=0))
    tau.append(np.pad(tau_, [[0,0],[0,0],[0,max_num_paths-num_paths]], constant_values=0))
# Let's now convert to uplink direction, by switing the receiver and transmitter
# dimensions
a = np.transpose(a, (2,3,0,1,4,5))
tau = np.transpose(tau, (1,0,2))

# Add a batch_size dimension
a = np.expand_dims(a, axis=0)
tau = np.expand_dims(tau, axis=0)

# Exchange the num_tx and batchsize dimensions
a = np.transpose(a, [3, 1, 2, 0, 4, 5, 6])
tau = np.transpose(tau, [2, 1, 0, 3])

# Remove CIRs that have no active link (i.e., a is all-zero)
p_link = np.sum(np.abs(a)**2, axis=(1,2,3,4,5,6))
a = a[p_link>0.,...]
tau = tau[p_link>0.,...]

print("Shape of a:", a.shape)
print("Shape of tau: ", tau.shape)
```
輸出:  
```python
Shape of a: (4727, 1, 16, 1, 4, 39, 14)
Shape of tau:  (4727, 1, 1, 39)
```

輸出說明:  
* 4277: 模擬完共有 4727 個 UE 的位置
* 1: UE 對應 1 個 BS
* 16: BS 的天線數量
* 1: 每筆資料只有一個 UE
* 4: UE 的天線數量
* 39: 路徑數量
* 14: 時間舉樣點

程式碼說明:  
* 每次模擬出來的 CIR 可能有 不一樣的 path 數
* 因此我們找出所有 CIR 中，最大的 path 數 `max_num_paths` ，並補零至統一長度
* `np.pad()` 說明：補零讓 `num_paths` 統一為 `max_num_paths`

#### 請注意，發射器和接收器的位置已顛倒，即發射器現在表示用戶設備（每個用戶設備有 4 根天線），接收器表示基地台（有 16 根天線）

#### Step 4. 現在讓我們定義一個資料產生器，它從 CIR 資料集中隨機取樣使用者裝置
##### 觀念
`class CIRGenerator`的功能是建立一個資料產生器，只要呼叫它就會產生 **multi-user uplink 的通道資料**:
* 從事先模擬好的 4727筆 CIR 資料中，挑出 `num_tx` 個 UE 的 CIR 資料
* 把這些 UE 的資料 stack 起來，讓我們後續能一次輸出整個 multi-user uplink 的 CIR

程式範例  

```python
class CIRGenerator:
    def __init__(self,
                 a,
                 tau,
                 num_tx):

        # Copy to tensorflow
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]

        self._num_tx = num_tx

    def __call__(self):

        # Generator implements an infinite loop that yields new random samples
        while True:
            # Sample 4 random users and stack them together
            idx,_,_ = tf.random.uniform_candidate_sampler(
                            tf.expand_dims(tf.range(self._dataset_size, dtype=tf.int64), axis=0),
                            num_true=self._dataset_size,
                            num_sampled=self._num_tx,
                            unique=True,
                            range_max=self._dataset_size)

            a = tf.gather(self._a, idx)
            tau = tf.gather(self._tau, idx)

            # Transpose to remove batch dimension
            a = tf.transpose(a, (3,1,2,0,4,5,6))
            tau = tf.transpose(tau, (2,1,0,3))

            # And remove batch-dimension
            a = tf.squeeze(a, axis=0)
            tau = tf.squeeze(tau, axis=0)

            yield a, tau

```
##### 第一步: 隨機選出多個 UE 的索引  
* `self._dataset_size = 4727`（假設你先傳進來的 a 有 4727 個 UE）
* `self._num_tx = 4`（表示你想模擬 multi-user uplink 中有 4 個 UE 同時傳送）
* 這段程式會從 4727 個 UE 中，隨機選出 4 個不同的 UE index → 存進 idx。

##### 第二步: 從dataset中提取這些UE的CIR  
* 對於 a：從 shape 為 (4727, 1, 16, 1, 4, 39, 14) 的張量中，選出 4 個 sample
* 所以新 shape 變成 (4, 1, 16, 1, 4, 39, 14)
    * 第 0 維：其實是 num_tx = 4，代表有 4 個不同 UE，每個都有一筆 CIR
    * 第 3 維：固定是 1，代表原本的每筆 CIR 都是「獨立的單一 UE」
    * 也就是說：你其實是擁有**4 筆 獨立的 UE CIR 資料**
* tau 同理，變成 (4, 1, 1, 39)
這就代表你現在有「4 個 UE → 傳送給同一個 BS」的 uplink CIR 資料

##### 第三步: 將原本「4個 UE 獨立的 CIR 資料」重新組裝成「一筆包含多個 UE 的 multi-user uplink CIR」格式
* `a = tf.transpose(a, (3,1,2,0,4,5,6))`: 把第 0 維和第 3 維互換
    * 以前是 4 筆單獨的 UE 資料，現在變成 1 筆包含 4 個 UE 的 multi-user uplink 資料
    * 新的 a.shape: (1, 1, 16, 4, 4, 39, 14)
* `tau = tf.transpose(tau, (2,1,0,3))`

##### 第四步: 移除 batch_size = 1 維度  
* `a = tf.squeeze(a, axis=0)`: 去除第0維
* `tau = tf.squeeze(tau, axis=0)`: 去除第0維

最後: 用`yield`輸出  
* 因為 `CIRGenerator` 無限循環的資料提供器，每次都要抽出一批新的 UE 資料
* 如果用 `return` ，執行一次就結束整個函數，下次不能再拿下一批資料了


#### Step 5. 把模擬的 CIR 資料，包裝成一個可以直接 plug-in 到 Sionna OFDMChannel
* 這一段，會使用 Sionna 提供的 `CIRDataset` class
* 它會把 CIR 包裝成一個「符合 Sionna 標準介面」的 channel model，供 [OFDMChannel](https://nvlabs.github.io/sionna/phy/api/channel.wireless.html#sionna.phy.channel.OFDMChannel) 使用
```python
batch_size = 20 # Must be the same for the BER simulations as CIRDataset returns fixed batch_size

# Init CIR generator
cir_generator = CIRGenerator(a,
                             tau,
                             num_tx)
# Initialises a channel model that can be directly used by OFDMChannel layer
channel_model = CIRDataset(cir_generator,
                           batch_size,
                           num_rx,
                           num_rx_ant,
                           num_tx,
                           num_tx_ant,
                           max_num_paths,
                           num_time_steps)
```

* `batch_size = 20`: 設定要一次從 CIRDataset 拿出多少筆資料。這個值要與後續的 BER 模擬使用的 batch 大小一致，否則會出現 shape 不相容錯誤
* `cir_generator = CIRGenerator(a, tau, num_tx)`: 建立一個資料生成器 CIRGenerator（前面你已經學過了），他會:
    *  從 a 和 tau（多個 UE 的 CIR）中，隨機抽出 num_tx 個 UE
    *  把它們對應的 CIR（通道係數與延遲）疊在一起，每次都產出一個新的 batch
* `channel_model = CIRDataset()`: 呼叫 Sionna 提供的 `CIRDataset` class，它會把 CIR 包裝成一個「符合 Sionna 標準介面」的 channel model，供 OFDMChannel 使用

#### 補充: 生成 `OFDMChannel` `h_freq`: 
* 在前面，我們已經完成了 `channel_model = CIRDataset(...)`這一步，代表已經用 ray-traced CIR 資料成功建立了一個可被 Sionna 認識的通道模型
* 接下來可以透過以下code，生成Channel frequency responses
* 用來給你每個 OFDM symbol、每個子載波的頻域產生channel coefficient

##### 第一步，準備 `resource_grid`

```python
from sionna.phy import ResourceGrid

resource_grid = ResourceGrid(num_ofdm_symbols=14,        # OFDM symbols per slot
                             num_subcarriers=72,         # subcarriers
                             subcarrier_spacing=30e3,    # 30 kHz
                             num_tx=4,                   # Number of UEs (TXs)
                             num_rx=1,                   # 1 base station
                             num_tx_ant=4,               # per UE
                             num_rx_ant=16,              # at BS
                             cp_length=72)               # CP samples
```

##### 第二步，建立頻域通道生成器 `GenerateOFDMChannel`

```python
from sionna.phy.channel import GenerateOFDMChannel

channel_freq_generator = GenerateOFDMChannel(channel_model=channel_model,
                                              resource_grid=resource_grid,
                                              normalize_channel=True)

```
* `Channel_model`: 是剛剛由 ray traced CIR 資料轉換而來的


##### 第三步，產生頻域通道響應 (Channel frequency responses)

```python
batch_size = 20
h_freq = channel_freq_generator(batch_size)
```
* input: `batch_size`
* `output`: `h_freq`
    * shape: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
 
***

## PUSCH Link-Level Simulations
* 除了丟到上述的 `OFDMChannel`之外，這段程式碼也能進行端對端 5G NR PUSCH uplink 傳輸的 BER 模擬
* 此模型針對符合 5G NR PUSCH 規範的多用戶 MIMO 上行通道運行 BER 模擬。

### Step 1. 你已經完成的部分（CIRDataset）
```python
channel_model = CIRDataset(
    cir_generator,
    batch_size,
    num_rx,
    num_rx_ant,
    num_tx,
    num_tx_ant,
    max_num_paths,
    num_time_steps
)
```

### Step 2. 
