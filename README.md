# High-Quality Novel View Synthesis from Noisy Images

## Installation
Clone this repository
```
git clone https://github.com/bo1230/NVS-from-NI.git 
cd NVS-from-NI
```

The code is tested with Python3.10, PyTorch==2.1.2+cu118 and cudatoolkit=11.8. To create a conda environment:
```
conda create –n NI python=3.10
conda activate NI
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```
For different platforms, the pytorch installation will probably be different.

## Datasets
Please refer to [IBRNet](https://github.com/googleinterns/IBRNet) and [LLFF](https://github.com/Fyusion/LLFF)for the dataset instruction.



## Training
Training with synthetic datasets
```
python train.py --config configs/synthetic_datasets/gain1/data2_benchflower_gain1.txt
```
Training with real datasets
```
python train.py --config configs/real_datasets/trash.txt
```
 




