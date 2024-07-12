# FOIL (ICML2024)
### This is an offical implementation of FOIL: [Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning](https://arxiv.org/abs/2406.09130). 

<div align="center">
    <img src="https://github.com/AdityaLab/FOIL/blob/main/Framework.png" width="500">
</div>

:triangular_flag_on_post:**News** (2024.06)  We are in the process of finalizing the code and will release it by July 14.
## Requirements
Dependencies can be installed using the following file: 
newtimelib_environment.yml
## Dataset
You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`
## Try out FOIL
Usecase
Run Raw Informer on ILI dataset with Pred_Len=4:
```bash
cd Informer-Raw
python ILI-Pred4.py 
```
Run Informer with FOIL on ILI dataset with Pred_Len=4:
```bash
cd Informer+FOIL
python ILI-Pred4-0.py 
python ILI-Pred4-1.py 
```
*  First Infer Envrionment; Second Learn Invariant Reperesentation
## Citation

If you find this repo useful, please cite our paper.

```
@article{liu2024time,
  title={Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning},
  author={Liu, Haoxin and Kamarthi, Harshavardhan and Kong, Lingkai and Zhao, Zhiyuan and Zhang, Chao and Prakash, B Aditya},
  journal={arXiv preprint arXiv:2406.09130},
  year={2024}
}
```

## Contact
If you have any questions or suggestions, feel free to contact:
hliu763@gatech.edu
## Acknowledgement

This library is constructed based on the following repos:
https://github.com/zhouhaoyi/Informer2020/
https://github.com/thuml/Time-Series-Library/
