# Generation of supervised training dataset for AI for diagnostic imaging by GAN: Datasets for supervised learning of breast ultrasound images by StyleGAN2

Norio Nakata, MD.

Division of Artificial Intelligence in Medicine, Jikei University, School of Medicine
### Table of contents
0. Image preparation
1. Training for image classification
     1-1. InceptionResNetV2 training with real images
     1-2. Generation of synthetic images by StyleGAN2
     1-3. Selection of synthetic images by trained model of real images
     1-4. InceptionResNetV2 training with synthetic images
2. InceptionRexNetV2 test and comparison of two models
3. Statistical analysis

![プレゼンテーション1](https://user-images.githubusercontent.com/47726033/111577690-1e00b700-87f6-11eb-8f5a-80e03be56180.jpg)
Figure1. Overall workflow

## 1. Image preparation
As a training image dataset for image generation in StyleGAN2
I prepared a 256x256 image in PNG format. Ideally, the number of images should be 10,000 or more, but if the number of images is 1000 or more, composition is possible. I used StyleGAN2 this time, but it seems that the successor version, StyleGAN2-ADA, can synthesize high-quality images with thousands of images. However, in our experience, in the case of the mammary gland ultrasound image used this time, StyleGAN2 was more suitable as supervised learning data for two-class classification than StyleGAN2-ADA, so StyleGAN2 was used this time. Information on this matter will be disclosed separately in the future.
## 2. StyleGAN2 implementation
Regarding the implementation of StyleGAN2, it conformed to the official GitHub.
The operating system used was Ubuntu 18.04.

// NVlabs/stylegan2: StyleGAN2 - Official TensorFlow Implementation
https://github.com/NVlabs/stylegan2

After implementing the Anaconnda environment first, T was based on the official GitHub. We used two NVIDIA GV100 GPUs. For StyleGAN, GPUs of Volta generation and above are recommended.

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

### Traing: run_train.py
Example: python run_training.py --num-gpus=2 --total-kimg=100000 --data-dir=datasets --config=config-e --dataset=benign-dataset --mirror-augment=true

### Monitor training:　Run tensorboard
Open another terminal and run tensorboard.
Reference: TensorBoard: TensorFlow's visualization toolkit
https://www.tensorflow.org/tensorboard

Example: python tensorboard --logdir results/00010-stylegan2-benign-dataset-2gpu-config-e

In our environment, training takes 3-4 days or more. Select the network with the lowest FID after the training and use it for image generation.

### Image generation: run_generator.py
Example 1:ython run_generator.py generate-images --network=results/00010-stylegan2-benign-dataset-2gpu-config-e/network-snapshot-012499.pkl --seeds=1-25000 --truncation-psi=1.6
Example 2: python run_generator.py generate-images --network=results/00011-stylegan2-cancer-dataset-2gpu-config-e/network-snapshot-009918.pkl --seeds=1-25000 --truncation-psi=1.6

## 4. InceptionResNetV2 implementation
Build another new Anaconda virtual environment as follows.

- cudatoolkit               10.1.243             
- cudnn                     7.6.5                   
- keras                     2.3.1                          
- keras-applications        1.0.8                 
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
2. 【Python】画像認識 - kerasで InceptionResNetV2をfine-tuningしてみる 【DeepLearning】 - PythonMania　（Japanese）
https://www.pythonmania.work/entry/2019/04/17/154153

## 5. Training with real images
## 6. Testing a model trained with　real images
## 7. Filtering the synthetic images
## 8. Traning with the synthetic images
## 9. Testing a model trained with synthetiv images
## 10. Comparision of real and synthetic case: Statistical analysis
