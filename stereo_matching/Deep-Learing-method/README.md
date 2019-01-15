# Deep Learning Method

We referenced a paper in 2018 named **"Pyramid Stereo Mathing Netwiork"**
and links below are useful to understand more about this great work:

* [arXiv link to paper](https://arxiv.org/abs/1803.08669)
* [Original github repo](https://github.com/JiaRenChang/PSMNet)
* [github repo of reimplementation](https://github.com/KinglittleQ/PSMNet)

The original github repo is full of bug and inconsistent in coding style, so we used the 
reimplementation one to get things work, however, the pretrained model can't be loaded 
for some reason, so we trained for about 700 epochs (batch size = 3), and we provide 
the pretrained model [here](https://drive.google.com/drive/u/0/my-drive)
and use our inference.py to predict disparity between left and right images by:

```
python3 inference.py \
  --left <left image path> \
  --right <right image path> \
  --model-path <path to trained model> \
  --save-path <path to the disparity image>
```

> Note: the code would generate **disp.py** too, which can be used to reconstruct 3D model (look corresponding directory for more details)

## left image
![left img](left.png)

## right image
![right img](right.png)

## disparity result
![disp img](disp.png)
