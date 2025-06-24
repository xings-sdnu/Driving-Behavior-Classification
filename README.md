# Driving-Behavior-Classification in PyTorch
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)Pytorch implementation for [Driving Behavior Classification Method Based on Fourier Transform Multimodal Fusion](https://ieeexplore.ieee.org/document/11047232/)(TITS)
## Requirements
PyTorch and Torchvision needs to be installed before running the scripts, together with `PIL` and `opencv` for data-preprocessing and `tqdm` for showing the training progress. PyTorch v1.1 is supported (using the new supported tensoboard); can work with ealier versions, but instead of using tensoboard, use tensoboardX.

```bash
pip install -r requirements.txt
```

or for a local installation

```bash
pip install --user -r requirements.txt
```

# Citation
Please cite our paper if you find the work useful:<br>

        @article{Sheng_TITS,
        author = {Xing Sheng, Jianrong Cao, Junzhe Zhang, Zhen Wang, Zongtao Duan},
        title = {Driving Behavior Classification Method Based on Fourier Transform Multimodal Fusion},
        journal = {IEEE Transcations on Intelligent Transportation Systems},
        volume = {},
        no = {},
        pages = {},
        year = {},
        doi = {https://doi.org/10.1109/TITS.2025.3579142}
        }
