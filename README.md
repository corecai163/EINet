# EINet
This is the open-source repository for the paper EINet: Point Cloud Completion via Extrapolation
and Interpolation

# Environment Setup

Install Pytorch 1.12.0 with Nvidia GPUs, check pytorch website https://pytorch.org/get-started/previous-versions/

Install required python packages using: pip install -r requirements.txt

Install pointnet2_ops_lib and Chamfer Distance in the extension Folder: sh install.sh


# Dataset
Please Check PoinTr (https://github.com/yuxumin/PoinTr/tree/master)

# Pretrained Models

[[Google Drive](https://drive.google.com/file/d/1S5WQVbWPjr7ip7pu0uugrO82ZT_f2WDL/view?usp=sharing)]

# Training/Testing
Please review the bash files (e.g., train.sh) and adjust the batch size and learning rates in the configuration (cfgs) files according to the number of GPUs in your system.

Check and modify the "test.sh" for testing.

# Citation
If our method and results are useful for your research, please consider citing:

```
@inproceedings{EINet,
    title={EINet: Point Cloud Completion via Extrapolation and Interpolation},
    author={Cai, Pingping and Zhang, Canyu and Shi, Lingjia and Wang, Lili and Imanpour, Nasrin and Wang, Song},
    booktitle={ECCV},
    year={2024},
}
```

# Acknowledgement
Our code is implemeted based on the PoinTr [https://github.com/yuxumin/PoinTr/tree/master]
