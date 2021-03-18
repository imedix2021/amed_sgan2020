# Generation of supervised training dataset for AI for diagnostic imaging by GAN: Datasets for supervised learning of breast ultrasound images by StyleGAN2

Norio Nakata, MD.

Division of Artificial Intelligence in Medicine, Jikei University, School of Medicine

1. Image preparation
2. StyleGAN2 implementation
3. Generation of images
4. InceptionResNetV2 implementation
5. Training with real images
6. Testing a model trained with　real images
7. Filtering the synthetic images
8. Traning with the synthetic images
9. Testing a model trained with synthetiv images
10. Comparision of real and synthetic case: Statistical analysis

![プレゼンテーション1](https://user-images.githubusercontent.com/47726033/111414603-9ba7c280-8723-11eb-9ec7-483dc213d760.jpg)
Figure1. Overall workflow

## 1. Image preparation
As a training image dataset for image generation in StyleGAN2
I prepared a 256x256 image in PNG format. Ideally, the number of images should be 10,000 or more, but if the number of images is 1000 or more, composition is possible. I used StyleGAN2 this time, but it seems that the successor version, StyleGAN2-ADA, can synthesize high-quality images with thousands of images. However, in our experience, in the case of the mammary gland ultrasound image used this time, StyleGAN2 was more suitable as supervised learning data for two-class classification than StyleGAN2-ADA, so StyleGAN2 was used this time. Information on this matter will be disclosed separately in the future.
## 2. StyleGAN2 implementation
Regarding the implementation of StyleGAN2, it conformed to the official GitHub.

// NVlabs/stylegan2: StyleGAN2 - Official TensorFlow Implementation
https://github.com/NVlabs/stylegan2

After implementing the Anaconnda environment first, T was based on the official GitHub. We used two NVIDIA GV100 GPUs. In our experience, it was difficult to operate StyleGAN2 on GPUs lower than V100.
After implementing the Anaconnda environment, the environment was constructed as follows.

- cudatoolkit               10.1.243             
- cudnn                     7.6.5       
- keras-applications        1.0.8                     
- keras-preprocessing       1.1.0                  
- numpy-base                1.19.1          
- python                    3.6.12                
- tensorboard               1.14.0         
- tensorflow                1.14.0        
- tensorflow-base           1.14.0         
- tensorflow-estimator      1.14.0             
- tensorflow-gpu            1.14.0         

## 3. Generation of synthetic images
### Run dataset_tool.py
Example:　python dataset_tool.py create_from_images ~/stylegan2/datasets/benign-dataset ~/BreastBenign

### Run train.py
Example: python run_training.py --num-gpus=2 --total-kimg=100000 --data-dir=datasets --config=config-e --dataset=benign-dataset --mirror-augment=true

## 4. InceptionResNetV2 implementation
Build another new Anaconda virtual environment as follows.

- cudatoolkit               10.1.243             
- cudnn                     7.6.5                   
- keras                     2.3.1                          
- keras-applications        1.0.8        After implementing another Anaconnda environment, the environment was constructed as follows.             
- keras-base                2.3.1                   
- keras-preprocessing       1.1.0                    
- matplotlib                3.1.3                    
- matplotlib-base           3.1.3             
- numpy                     1.18.1             
- numpy-base                1.18.1           
- pandas                    1.0.1            
- pillow                    7.0.0             
- python                    3.7.6          
- scikit-learn              0.22.1        
- scipy                     1.4.1           
- tensorboard               2.2.1          
- tensorboard-plugin-wit    1.6.0        
- tensorflow                2.2.0          
- tensorflow-base           2.2.0           
- tensorflow-estimator      2.2.0             
- tensorflow-gpu            2.2.0      
     
References:
1. Keras Applications
https://keras.io/api/applications/
2.【Python】画像認識 - kerasで InceptionResNetV2をfine-tuningしてみる 【DeepLearning】 - PythonMania　（Japanese）
https://www.pythonmania.work/entry/2019/04/17/154153



## 5. Training with real images
## 6. Testing a model trained with　real images
## 7. Filtering the synthetic images
## 8. Traning with the synthetic images
## 9. Testing a model trained with synthetiv images
## 10. Comparision of real and synthetic case: Statistical analysis
