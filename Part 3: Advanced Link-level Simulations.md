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




