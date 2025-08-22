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

# Sample batch_size random user positions from the radio map
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

接著根據上述條件，從 Radio Map 中**篩選出合法的 UE 位置候選點**




