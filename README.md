# real_time_segmentation
entropy guided feature extraction for real time semantic segmentation

| Model         | Pixel Acc. | 
| :------------ |:---------------:| 
| Baseline      | 46.23%    |   
| EntropyFeat         | 46.96%    |  

# Architecture
<img src="https://github.com/lusinlu/real_time_segmentation/blob/main/architecture.png" width="1000" height="500">

for running cityscapes
python3 main.py --cuda --dataset cityscapes --data_path ../../datasets/cityscapes/ --classes 20 -ignore_label 0
