### Investigating Multi-Rater Segmentation Annotations of Optic Disc and Cup
------

<img src="https://github.com/Guoxt/RPRC/blob/master/image0.png" alt="Image Alt Text" style="width:400px; height:auto;">

------
### Introduction 

<div style="text-align: justify;"> 

##### The segmentation of the optic disc (OD) and optic cup (OC) from fundus images suffers from rater variation due to differences in raters' expertise and image blurriness. In clinical, we can fuse the annotations of multiple raters to reduce rater-related biases through methods such as mean voting. However, these methods ignores the unique preferences of each rater. In this paper, we propose a novel neural network framework to jointly learn rater calibration and preference for multiple annotations of OD and OC segmentation, which consists of two main parts. In the first part, we employ an encoder-decoder network to learn the annotation variation among multiple raters and produce calibrated image segmentation. Further, we are the first to propose a multi-annotation smoothing network (MSNet), which can effectively remove high-frequency components in calibration predictions. In the second part, we represent different raters with specific codes, which are used as parameters of the model and optimized during training. In this way, we can achieve modeling different rater preferences under a single network, and the learned rater codes can represent differences in preference patterns among different raters. Our experiments show that our framework outperforms a range of state-of-the-art (SOTA) methods. 

</div>

------
### Framework
------

<img src="https://github.com/Guoxt/RPRC/blob/master/image1.png" alt="Image Alt Text" style="width:1000px; height:auto;">

------
### Run MRNet Code

1. train

```python main.py```                              # set '--phase' as train cal
```python main_LRP.py```                          # set '--phase' as train apm

2. test

```python test.py```                        # set '--phase' as test


### Dataset
1. RIGA benchmark: you can access to this download [link](https://pan.baidu.com/s/1CJzY6WYJfyLwDEKGo_D2IQ), or [Google Drive](https://drive.google.com/drive/folders/1Oe9qcuV0gJQSe7BKw1RX_O1dGwenIx9i?usp=sharing). 
```
