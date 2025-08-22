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
* TX: 基地台 ； RX: UE
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
