## Asymmetric Region Denoising and Rotation Equivariant for Image Reflection Symmetry Detection

<p align="center">
Dongfu Yin, Rourou Su*, Cong Zhao, Fei Yu
</p>

<p align="center">
    <a href="">[paper]</a>
</p>

Implementation of *Asymmetric Region Denoising and Rotation Equivariant for Image Reflection Symmetry Detection*.

### Environment
```
    conda create --name ARDNet python=3.7
    conda activate ARDNet
    conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=11.0 -c pytorch
    conda install -c conda-forge matplotlib
    pip install albumentations==0.5.2 shapely opencv-python tqdm e2cnn mmcv
```

### Datasets and weights
- download DENDI、LDRS、NYU、SDRW [sym_datasets](https://pan.baidu.com/s/1m1iKqmHeVzMInStrwGvatQ?pwd=3ew2)
- download trained [weights](https://pan.baidu.com/s/1ssS9YYIM57gvg45Zfq4kyg?pwd=bpuu): ARDNet(ours), pre-trained ReResNet50(D8)

```
.
├── sym_datasets
│   └── DENDI
│   └── LDRS
│   └── NYU
│   └── SDRW
├── weights
│   ├── v_ardnet_best_checkpoint.pt
│   └── re_resnet50_custom_d8_batch_512.pth
├── (...) 
└── train.py
```


### demo(visualization)

```
python demo.py --ver ardnet -rot 0 -eq --get_theta 10
```

### test

```
python train.py --ver ardnet -t -rot 0 -eq -wf --get_theta 10
```

### train
```
python train.py --ver ardnet -tlw 0.01 --get_theta 10 -rot 0 -eq --lr 0.001 -wf
```
- you can change the --ver to you version, such as --ver mynet
- you can change the learning rate using --lr, such as --lr 0.01


### References
- DENDI [link](https://github.com/ahyunSeo/DENDI)

