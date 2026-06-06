# STA-MRI: Toward Physics-Aware MRI ImageSuper-Resolution

## 💡 Summary
we propose a novel 2D Gaussian splatting-based MRI super-resolution framework that accommodates dynamically varying input resolutions. We further introduce a prior-aware Gaussian parameterization module to enhance structural fidelity and a physics-constrained signal modeling module to ensure biophysically plausible intensity reconstruction. 

## 💡 Motivation and Framework
![Motivation](assets/motivation.png)
Illustration of the trade-off between spatial resolution and signal-to-noise ratio (SNR) under a simulated ultra-low MRI system(64 mT). In the high-resolution but low-SNR setting, severe noise leads to fragmented and discontinuous anatomical structures, as highlighted in the yellow boxes. The balanced regime produces more coherent and struc-turally consistent reconstructions, closely matching the HR reference. In the low-resolution high-SNR setting, structures appear over-smoothed, and fine anatomical details are lost due to partial volume effects.

## 📃 Dependencies and Installation
python 3.10
pytorch 1.10.0
```bash
pip install requirements.txt
git clone https://github.com/XingtongGe/gsplat.git
cd gsplat
pip install -e .[dev]
```

## Get Started
### Datasts
[Simulated datasets from fastMRI](https://drive.google.com/file/d/1m969yGjzLv2ydrmlCEdf4hP-XH009W3X/view?usp=drive_link),
[Simulated datasets from IXI ](https://drive.google.com/file/d/1GAZ_CL2L6CR7DqcR5TGYwc1CNHsG2T1u/view?usp=drive_link),
[Real-world 3T-5T dataset](https://drive.google.com/file/d/1nCv-zUipGTQsS5Bd1VcH3TsnT0ckXEAg/view?usp=drive_link),
[Real 64mT-3T paired dataset](https://zenodo.org/records/15862148)

### Pretrained model
- Dynamic-resolution experiment on simulated dataset: [STAmodel](https://drive.google.com/file/d/1_TlA5dnYrowVsQP16EpQtDgNSh-_5V3L/view?usp=drive_link)
- Dynamic-resolution experiment on fastMRI dataset: [Scale_4](https://drive.google.com/file/d/1lnflIf03W33XvrB-FZpMnH-ZqkOamZIj/view?usp=drive_link),[Scale_5](https://drive.google.com/file/d/1RNbxic12M1FKTxWV8Vl4HD-XL10mGbNo/view?usp=drive_link)
[Scale_6.4](https://drive.google.com/file/d/1a676KIrhf0RNcK76f7sWFeKOcpA5sgRc/view?usp=drive_link)
- Dynamic-resolution experiment on real dataset:[STAmodel_3t5t_finetune](https://drive.google.com/file/d/1kGM5ZfN2X9aZsExG182b3J0RYhUidXt7/view?usp=drive_link)
- Static-resolution experiment : [STAmodel_static_experiment](https://drive.google.com/file/d/15X4U_0wSmW8dtETTQPSBZmUlXCzU4nQC/view?usp=drive_link)
- Once downloaded, place the model in the designated folder, and you’ll be ready to perform inference.

### Inference
Place all pretrained weights in "save" folders firstly. Here are  example commands for inference
```bash
python test.py --config configs/test/test_dynamic_train_mri_seg_mask.yaml --checkpoint save/dynamic_experiment/STAmodel.pth
python test.py --config configs/test/test_3t5t_finetune.yaml --checkpoint save/dynamic_experiment/STAmodel_3t5t_finetune.pth
python test_fastMRI.py --test_config ./configs/test/test_fastmri_static_k_4.0.yaml --checkpoint save/dynamic_experiment/checkpoint_best_scale4.pth
python test.py --config configs/test/test_static.yaml --checkpoint save/static_experiment/STAmodel_static_experiment.pth
```
## Visual Examples

![VISUAL](assets/motivation.png)

## ✉️ License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

### Citation

If you are interested in the following work, please cite the following paper.

```




```