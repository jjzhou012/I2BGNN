# I2BGNN

This is a PYG implementation of I2BGNN, as described in the following:
> Identity inference on blockchain using graph neural network


## Requirements
For hardware configuration, the experiments are conducted at Ubuntu 18.04.5 LTS with the Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz, and NVIDIA Tesla V100S GPU (with 40GB memory each).
For software configuration, all model are implemented in
- Python 3.7
- Pytorch-Geometric 2.0.3
- Pytorch 1.8.0
- Scikit-learn 0.24.1
- CUDA 10.2


## Data
Download data from this [link](https://zjuteducn-my.sharepoint.com/:f:/g/personal/jjzhou_zjut_edu_cn/Ei03RKWmCRVMnjyYdUbHLSwBpa_Y4b_ZULfrre8348uebQ?e=NEx4Eg) and place it under the 'data/eth/' path.

## Usage
Execute the following bash commands in the same directory where the code resides:
  ```bash
  $ python main.py -l p --hop 2 -ess Volume -layer 2 --pooling max --hidden_dim 128 --batch_size 32 --lr 0.001 --dropout 0.2 -undir 1 -which_ew Volume
  ```
More parameter settings can be found in 'utils/parameters.py'.


## Citation

If you find this work useful, please cite the following:

```bib
@inproceedings{shen2021identity,
  title={Identity inference on blockchain using graph neural network},
  author={Shen, Jie and Zhou, Jiajun and Xie, Yunyi and Yu, Shanqing and Xuan, Qi},
  booktitle={International Conference on Blockchain and Trustworthy Systems},
  pages={3--17},
  year={2021},
  organization={Springer}
}
```

