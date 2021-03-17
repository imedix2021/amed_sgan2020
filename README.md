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

## 3. Generation of images
## 4. InceptionResNetV2 implementation
## 5. Training with real images
## 6. Testing a model trained with　real images
## 7. Filtering the synthetic images
## 8. Traning with the synthetic images
## 9. Testing a model trained with synthetiv images
## 10. Comparision of real and synthetic case: Statistical analysis
