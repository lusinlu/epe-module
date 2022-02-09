# EPE Module
[ICIP 2022] Official implementation of the [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Abrahamyan_Bias_Loss_for_Mobile_Neural_Networks_ICCV_2021_paper.pdf) "Entropy-Based Feature Extraction For Real-Time Semantic Segmentation".

## Usage (test)
Pretrained RTEffNet+EPE model is available from [Google Drive](https://drive.google.com/file/d/12H8WmfGOX4cZ9jeFPAo6aHU7LzD7HtBE/view?usp=sharing). For the testing of the model using the Cityscapes validation set run the following command:

`python test.py --data_path path/to/validation/set --cuda --weights path/to/downloaded/weight `

## Usage (train)
To train the RTEffNet+EPE module on Cityscapes dataset run the following command:

` python main.py --dataset cityscapes --data_path ../../datasets/cityscapes/ --cuda`

## Architecture
<img src="https://github.com/lusinlu/real_time_segmentation/blob/main/architecture.png" width="1100" height="500">


## Citation
If you find the code useful for your research, please consider citing our works

```
@article{abrahamyanepe,
  title={Entropy-Based Feature Extraction For Real-Time Semantic Segmentation},
  author={Lusine, Abrahamyan and Nikos, Deligiannis},
  journal={Proceedings of the IEEE/CVF International Conference on Image Processing (ICIP)},
  publisher = {IEEE},
  year={2022}
}
```

## Acknowledgement
Code for EfficientNet is borrowed from [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) and for the Cityscapes loader from [TORCHVISION.DATASETS](https://pytorch.org/vision/0.8/datasets.html) and [pytorch-semantic-segmentation](https://github.com/zijundeng/pytorch-semantic-segmentation). 
