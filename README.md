# 🛰️ Packet Routing via Multi-Agent Reinforcement Learning (MARL)

This is a demo project that implements **packet routing** via **MARL** using the [Xuance](https://github.com/agi-brain/xuance) framework.

---


## 🧰 Prerequisites

Make sure the following tools are installed on your system:

1. **[PyCharm Professional](https://www.jetbrains.com/zh-cn/pycharm/download/)** (latest version recommended)  
2. **[Anaconda](https://www.anaconda.com/download/success)** (latest version recommended)  
3. **[Xuance](https://github.com/agi-brain/xuance)** (latest version)

---

## 🚀 Installation Steps
### 1. Clone Xuance Framework

```bash
git clone https://github.com/agi-brain/xuance.git
cd xuance
```

Xuance is a general-purpose multi-agent reinforcement learning framework.
👉 For more details, visit the [official documentation](https://xuance.readthedocs.io/zh/latest/). 


### 2. Create and Set Up a Conda Environment

We recommend Python 3.8.
```bash
conda create -n xuance_env python=3.8
conda activate xuance_env
```

Install required dependencies:
```bash
conda install mpi4py
pip install xuance
```


### 3. Run the Default Demo (Optional Check)
Run the default MARL demo to verify the setup:
```bash
python benchmark_marl.py
```
If it runs successfully, your environment is ready.


### 4. Add Custom Packet Routing Environment
Copy the following custom files to the appropriate directories:
```bash
- Copy `benchmark_marl.py` to the project root directory.
- Copy `packet_routing.py` to: ./xuance/environment/multi_agent_env/
- Copy `satellite_network.py` to: ./xuance/environment/multi_agent_env/
- Copy `__init__.py` to: ./xuance/environment/multi_agent_env/
- Copy `packet_routing.yaml` to: ./xuance/configs/mappo/
```
Make sure the file names and paths are exactly as shown.


### 5. Run Training with Custom Environment
Execute the training script using PyTorch:
```bash
python benchmark_marl.py
```

### 6. Visualize Training with TensorBoard
Xuance integrates with TensorBoard for real-time training visualization:
```bash
tensorboard --logdir=./logs/mappo/torch
```
Then open the URL provided in your terminal (usually http://localhost:6006) to view the training metrics in your browser.


---
## 📌 Notes
- This demo uses the MAPPO algorithm.
- Xuance supports various MARL algorithms like MAPPO, QMIX, IPPO, etc.
- For further configuration and tuning, refer to the Xuance config docs.


---
## 📂 Project Structure (Custom Files)
```bash
xuance/
│
├── benchmark_marl.py                 # Entry script for training
├── xuance/
│   └── environment/
│       └── multi_agent_env/
│           └── packet_routing.py    # Custom environment
│           └── satellite_network.py # Cooperate with custom environment
│           └── __init__.py          # For registering the environment
│
└── xuance/
    └── configs/
        └── mappo/
            └── packet_routing.yaml  # Custom config for MAPPO
```


---
## 📜 License
This project inherits the license of Xuance. See xuance/LICENSE for more details.

