
#  Information transmission: Inferring change area from change moment in time series remote sensing image

## Jialu Li， Chen Wu，Meiqi Hu， and Haonan Guo
 


This is an offical implementation of CAIM-Net framework in our  ISPRS JP&RS 2026 paper:Information transmission: Inferring change area from change moment in time series remote sensing image.
 (https://www.sciencedirect.com/science/article/pii/S0924271625004228)

![image](https://github.com/user-attachments/assets/8b5612cb-ee5f-4c86-bea6-f18b6aa22d4d)

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
Two datasets, UTRNet dataset and DynamicEarthNet dataset, are used for experiments, Please down them in the following way.<br>

UTRNet dataset: https://github.com/thebinyang/UTRNet<br>

The DynamicEarthNet dataset original image can be downloaded： https://mediatum.ub.tum.de/1650201<br>

The UTRNet dataset used in our paper after processing can be download: https://pan.baidu.com/s/1kI6bHRf11t0qvUefqpClKA?pwd=ssye 提取码: ssye

The DynamicEarthNet dataset used in our paper after processing can be download: https://pan.baidu.com/s/1zatEQIYVgMrcy1IDUnVrBw 提取码：1442 

### Training and Testing Multi-RLD-Net 

<pre><code id="copy-text">python Train_Multi_RLD_Net.py
python Test_Multi_RLD_Net.py</code></pre>

![image](https://github.com/user-attachments/assets/1b6439ef-bd5b-4ad4-89ee-50aad0ce3006)


### Test our trained model results
you can directly test our model by our provided training weights in best_model. 

<pre><code id="copy-text">save_path = best_model + 'Dynamic_Multi_RLD_Net.pth'</code></pre>

### Training and Testing RLD-Net on UTRNet 

<pre><code id="copy-text">python Train_UTRNet.py
python Test_UTRNet.py</code></pre>

<img width="664" height="413" alt="image" src="https://github.com/user-attachments/assets/9ce781f9-ef18-45c3-b4f2-40a8f45ec1e6" />



### Test our trained model results
you can directly test our model by our provided training weights in best_model. 

<pre><code id="copy-text">save_path = best_model + 'UTRNet_RLD_Net.pth'</code></pre>

### Citation 
if you use this code for your research, please cite our papers. 

<pre><code id="copy-text">@article{Li2024Multi-RLD-Net,
    title = {Using difference features effectively: A multi-task network for exploring change areas and change moments in time series remote sensing images},
    author = {Jialu Li and Chen Wu},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {218},
    pages = {487-505},
    year = {2024},
    issn = {0924-2716},
    doi = {https://doi.org/DOI10.1016/j.isprsjprs.2024.09.029}
}</code></pre>



