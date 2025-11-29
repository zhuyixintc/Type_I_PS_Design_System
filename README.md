# Type_I_PS_Design_System
We couple a graph neural network-based surrogate for excited-state properties estimation with a DQN-based fragment assembly system that search molecules toward Type I PSs.

<p align="left">
<img width="330" height="294" alt="image" src="https://github.com/user-attachments/assets/c2fc7709-bd66-4838-89a6-4d3c8a1ac6ce" />
</p>

## Requirements
- Pyhton=3.9.7
- CUDA=11.2
- TensorFlow=2.10.1
- RDKit=2022.03.5

## Installation
- conda create -n typecips python=3.9.7 -y
- conda activate typecips
- conda install -c conda-forge rdkit=2022.03.5 -y
- pip install tensorflow==2.10.1

## Code Structure
```text
Type_I_PS_Design_System/
├─ models/
│  ├─ agent.py
│  ├─ buffer.py
│  ├─ data_load.py
│  ├─ environment.py
│  ├─ features.py
│  ├─ molecular_generation.py
│  ├─ mpnn_model.py
│  ├─ predictor.py
│  ├─ train_mpnn.py
│  └─ utils.py
├─ run_dqn.py
├─ run_mpnn.py
└─ README.md
```
## Workflow
- Your own data and define your targets
- python run_mpnn.py
- python run_dqn.py

## UI download
- Link: https://pan.baidu.com/s/1kIcsqaMkwo0-yIOZURK7dw
- Password: 7777 

## Acknowledgement
This work is partially built on MolDQN and uses a subset of xTB-ML-data. We deeply grateful to the authors for making their code and data publicly available.

The dataset and trained models will be made available on GitHub upon publication.
