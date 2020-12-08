# IIM - Crowd Localization

---

This repo is the official implementation of [paper](): **Learning Independent Instance Maps for Crowd Localization**. The code is developed based on [C3F](https://github.com/gjy3035/C-3-Framework). 

# Progress
- [x]  NWPU
  - [x] processing data
  - [x] training parameters
  - [x] HR models
  - [ ] VGG+FPN models
- [ ] UCF-QNRF
- [ ] SHT A
- [ ] SHT B
- [ ] FDST 

# Getting Started

## Preparation
- Prerequisites
    - Python 3.x
    - Pytorch 1.x: http://pytorch.org .
    - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```.
-  Preparation
    - Clone this repo in the directory (```Root/IIM```):
    - Download NWPU-Crowd dataset from this [link](https://www.crowdbenchmark.com/nwpucrowd.html). 
    - Unzip ```*zip``` files in turns and place ```images_part*``` into the same folder (```Root/ProcessedData/NWPU/images```). 
    - Download the processing labels and val gt file from this [link](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/EliCeOckaZVBgez6n8ZWvr4BNdwPauFJgbm88MGhHid25w?e=rtogwc). Place them into ```Root/ProcessedData/NWPU/masks``` and ```Root/ProcessedData/NWPU```, respectively.
    - Download the pre-trained HR models from this [link](https://onedrive.live.com/?authkey=%21AKvqI6pBZlifgJk&cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21116&parId=F7FD0B7F26543CEB%21105&action=locate). More details are availble at [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) and  [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification).
  - Finally, the folder tree is below:
 ```
    -- ProcessedData
		|-- NWPU
			|-- images
			|   |-- 0001.jpg
			|   |-- 0002.jpg
			|   |-- ...
			|   |-- 5109.jpg
			|-- masks
			|   |-- 0001.png
			|   |-- 0002.png
			|   |-- ...
			|   |-- 3609.png
			|-- train.txt
			|-- val.txt
			|-- test.txt
			|-- val_gt_loc.txt
	-- PretrainedModels
	  |-- hrnetv2_w48_imagenet_pretrained.pth
	-- IIM
	  |-- datasets
	  |-- misc
	  |-- ...
 ```

## Training
- run ```python train.py```.
- run ```tensorboard --logdir=exp --port=6006```.
- The validtion records are shown as follows:
   ![val_curve](./figures/curve.png)
- The sub images are the input image, GT, prediction map,localization result, and pixel-level threshold, respectively:
   ![val_curve](./figures/vis.png)


## Testing

ToDo.



# Performance on the validation set

The results (F1, Pre., Rec.) and [pre-trained models](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/EliCeOckaZVBgez6n8ZWvr4BNdwPauFJgbm88MGhHid25w?e=rtogwc) on NWPU val set, UCF-QNRF, SHT A, SHT B, and FDST:

|   Method   |  NWPU val  |  UCF-QNRF  |  SHT A  |  SHT B  |  FDST |
|------------|-------|-------|--------|--------|--------|
| Paper:  HRNet [1]   | 80.2/84.1/76.6| - | -  |  -  |
| Paper:  VGG+FPN [2,3]| - | - | - |  - |
| This Repo:  HRNet [1]   | - | - | -  |  -  |
| This Repo:  VGG+FPN [2,3]| -| - | - |  - |
1. Deep High-Resolution Representation Learning for Visual Recognition, T-PAMI, 2019.
2. Very Deep Convolutional Networks for Large-scale Image Recognition, arXiv, 2014.
3. Feature Pyramid Networks for Object Detection, CVPR, 2017. 

About the leaderboard on the test set, please visit [Crowd benchmark](https://www.crowdbenchmark.com/nwpucrowdloc.html).  Our submissions are the [IIM(HRNet)](https://www.crowdbenchmark.com/resultldetail.html?rid=11) and [IIM (VGG16)](https://www.crowdbenchmark.com/resultldetail.html?rid=10).



# Video Demo

We test the pretrained HR Net model on the NWPU dataset in a real-world subway scene. Please visit [bilibili](https://www.bilibili.com/video/BV1K541157MK) or [YouTube](https://www.youtube.com/watch?v=GqOMgjUkbsI) to watch the video demonstration.
![val_curve](./figures/vid.png)
# Citation
If you find this project is useful for your research, please cite:
```
@article{gao2020learning,
  title={Learning Independent Instance Maps for Crowd Localization},
  author={Gao, Junyu and Han, Tao and Yuan, Yuan and Wang, Qi},
  journal={arXiv preprint arXiv:20XX},
  year={2020}
}
```

Our code borrows a lot from the C^3 Framework, you may cite:
```
@article{gao2019c,
  title={C$^3$ Framework: An Open-source PyTorch Code for Crowd Counting},
  author={Gao, Junyu and Lin, Wei and Zhao, Bin and Wang, Dong and Gao, Chenyu and Wen, Jun},
  journal={arXiv preprint arXiv:1907.02724},
  year={2019}
}
```
If you use pre-trained models in this repo (HR Net, VGG, and FPN), please cite them. 


