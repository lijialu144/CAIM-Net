
#  Information transmission: Inferring change area from change moment in time series remote sensing image

## Jialu Li， Chen Wu，Meiqi Hu， and Haonan Guo
 


This is an offical implementation of CAIM-Net framework in our  ISPRS JP&RS 2026 paper:Information transmission: Inferring change area from change moment in time series remote sensing image.
 (https://www.sciencedirect.com/science/article/pii/S0924271625004228)

![image](https://ars.els-cdn.com/content/image/1-s2.0-S0924271625004228-gr2.jpg)

## Get started

### Requirements
please download the following key python packages in advance.

<pre><code id="copy-text">python==3.7.16
pytorch==1.12.1
scikit-learn==1.0.2
scikit-image==0.19.3
imageio=2.31.2
numpy==1.21.5
tqdm==4.66.6</code></pre>

### Dataset 
Two datasets, DynamicEarthNet dataset and SpaceNet7 dataset, are used for experiments, Please down them in the following way.<br>

The DynamicEarthNet dataset original image can be downloaded： https://mediatum.ub.tum.de/1650201<br>

The SpaceNet7 dataset original image can be downloaded： https://gitcode.com/Resource-Bundle-Collection/6e3fe/tree/main

The DynamicEarthNet dataset used in our paper after processing can be download: https://pan.baidu.com/s/1zatEQIYVgMrcy1IDUnVrBw 提取码：1442 

The SSpaceNet7 dataset used in our paper after processing can be download: https://pan.baidu.com/s/1zatEQIYVgMrcy1IDUnVrBw 提取码：1442 

### Data Processing
Before Training the model, you need to run the Fsplit, create_xys, and custom files in the data_processing folder.


### Training and Testing CAIM-Net on DynamicEarthNet dataset

<pre><code id="copy-text">python Train_Dynamic.py 
Python Test_Dynamic.py</code></pre>

![image](https://ars.els-cdn.com/content/image/1-s2.0-S0924271625004228-fx1.jpg)


### Test our trained model results
you can directly test our model by our provided training weights in best_model. 

<pre><code id="copy-text">save_path = best_model + 'Pretrained_model_Dynamic.pth'</code></pre>

### Training and Testing CAIM-Net on SpaceNet7 dataset

<pre><code id="copy-text">python Train_SpaceNet7c.py 
Python Test_SpaceNet7.py</code></pre>

![image](https://ars.els-cdn.com/content/image/1-s2.0-S0924271625004228-fx2.jpg)

### Test our trained model results
you can directly test our model by our provided training weights in best_model. 

<pre><code id="copy-text">save_path = best_model + 'Pretrained_model_SpaceNet7.pth'</code></pre>

### Citation 
if you use this code for your research, please cite our papers. 

<pre><code id="copy-text">Li, Jialu, et al. "Information transmission: Inferring change area from change moment in time series remote sensing images." ISPRS Journal of Photogrammetry and Remote Sensing 231 (2026): 266-287.</code></pre>




