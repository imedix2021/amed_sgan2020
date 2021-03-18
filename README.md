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

## 0. Image preparation
First, the images are divided into "Train" and "Test".
150 benign and malignant images for the "Test", and the others are randomly selected and classified for "Train". The image format uses BMP for classification and PNG for image generation. All image sizes are 256x256.
Ideally, the number of images should be 10,000 or more, but if the number of images is 1000 or more, composition is possible. I used StyleGAN2 this time, but it seems that the successor version, StyleGAN2-ADA, can synthesize high-quality images with thousands of images. However, in our experience, in the case of the mammary gland ultrasound image used this time, StyleGAN2 was more suitable as supervised learning data for two-class classification than StyleGAN2-ADA, so StyleGAN2 was used this time. Information on this matter will be disclosed separately in the future.
## 1. Training for image classification
### 1-1. InceptionResNetV2 training with real images
Build the Anaconda virtual environment as follows.
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

Download all the Codes for this GituHub.
Copy the selected 150 images to "BreastBenign" and "BreastMalignancy" in the "Test" folder in the "hikakudata" folder in the "real" folder.
Then zip the hikakudata folder.

Run classification training: 
```
python IRV2_755_real_breastUS.py
```
Test after training.`
```
python IRV2_755_real_breastUS_test.py
```
The test result is output as follows.
```
Real 
2x2_IncRenNetV2_775
Benign
[[138 12]
[ 18 132]]
Benign precision: 0.9166666666666666 ( 132 / 144.0 )
Benign recall(sensitivity): 0.88 ( 132 / 150.0 )
Benign specificity: 0.92 ( 138 / 150.0 )
Benign f_1 score: 0.8979591836734694 ( 1.6133333333333333 / 1.7966666666666666 )

Malignancy
[[132 18]
[ 12 138]]
Malig precision: 0.8846153846153846 ( 138 / 156.0 )
Malig recall(sensitivity): 0.92 ( 138 / 150.0 )
Malig specificity: 0.88 ( 132 / 150.0 )
Malig f_1 score: 0.9019607843137256 ( 1.6276923076923078 / 1.8046153846153845 )

Accuracy: 0.9 ( 270 / 300.0 )

precision recall f1-score support

Benign 0.917 0.880 0.898 150
Malignant 0.885 0.920 0.902 150

accuracy 0.900 300
macro avg 0.901 0.900 0.900 300
weighted avg 0.901 0.900 0.900 300

300/300 [==============================] - 1s 4ms/step
AUC: 0.9650666666666666
```    
References:
1. Keras Applications
https://keras.io/api/applications/
2. 【Python】画像認識 - kerasで InceptionResNetV2をfine-tuningしてみる 【DeepLearning】 - PythonMania　（Japanese）
https://www.pythonmania.work/entry/2019/04/17/154153
## 1-2. Generation of synthetic images by StyleGAN2
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
- 
### Run dataset_tool.py
Example:　
```    
python dataset_tool.py create_from_images ~/stylegan2/datasets/benign-dataset ~/BreastBenign
```    
### Traing: run_train.py
Example: 
```    
python run_training.py --num-gpus=2 --total-kimg=100000 --data-dir=datasets --config=config-e --dataset=benign-dataset --mirror-augment=true
```    
### Monitor training:　Run tensorboard
Open another terminal and run tensorboard.
Reference: TensorBoard: TensorFlow's visualization toolkit
https://www.tensorflow.org/tensorboard

Example:
```    
python tensorboard --logdir results/00010-stylegan2-benign-dataset-2gpu-config-e
```    
In our environment, training takes 3-4 days or more. Select the network with the lowest FID after the training and use it for image generation.

### Image generation: run_generator.py
Generates 25,000 images for both benign and malignant.
Example :
```    
python run_generator.py generate-images --network=results/00010-stylegan2-benign-dataset-2gpu-config-e/network-snapshot-012499.pkl --seeds=1-25000 --truncation-psi=1.6
python run_generator.py generate-images --network=results/00011-stylegan2-cancer-dataset-2gpu-config-e/network-snapshot-009918.pkl --seeds=1-25000 --truncation-psi=1.6
```    
## 1-3. Selection of synthetic images by trained model of real images
Copy the created 25,000 images to the / sg2t16_28000 / sgan_out / BreastBenign and BreastMalignancy folders, respectively.
Then zip the sgan_out foldger.
Run the select the images of sg2t16_28000 folder.
```
python IRNV2_755_28000_breast_select_all.py
```


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
