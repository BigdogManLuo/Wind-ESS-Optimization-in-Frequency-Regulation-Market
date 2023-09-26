# Wind-ESS-Optimization-in-Frequency-Regulation-Market
![](https://img.shields.io/badge/python-3.9.13-brightgreen)  ![](https://img.shields.io/badge/numba-0.55.1-blue)  ![](https://img.shields.io/badge/license-MIT-orange) <br>
Code implementation of [Flexible Coordination of Wind Generators and Energy Storages in joint Energy and Frequency Regulation Market](https://ieeexplore.ieee.org/document/10140535), where we elaborate on our methodology, implementation details, and experimental results.


## Requirements
- Python 3.6+
- Anaconda 23.1.0
- Numba 0.55.1
  
## Installation

1.Clone the repository

```sh
git clone https://github.com/BigdogManLuo/Wind-ESS-Optimization-in-Frequency-Regulation-Market.git
```
2.Install the required dependencies:
```sh
pip install numba
...
```

# Reference
If you find this work useful and use it in your research, please consider citing our paper:
```bibtex
@misc{@INPROCEEDINGS{10140535,
  author={Pu, Chuanqing and Xiang, Yue and Fan, Feilong and Huan, Jiafei and Deng, Li and Liu, Junyong},
  booktitle={2023 Panda Forum on Power and Energy (PandaFPE)}, 
  title={Flexible Coordination of Wind Generators and Energy Storages in Joint Energy and Frequency Regulation Market}, 
  year={2023},
  volume={},
  number={},
  pages={1926-1931},
  doi={10.1109/PandaFPE57779.2023.10140535}}
}
```

## Usage

Run these numerical experiment as follows.
```sh
cd base
python 01_DP.py
python 02_RealTimeOpt.py
```


## Acknowledgement

Thanks to Feilong Fan and Yue Xiang for their guidance in this work.







