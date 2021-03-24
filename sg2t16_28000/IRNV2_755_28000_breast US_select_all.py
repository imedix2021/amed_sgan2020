from __future__ import print_function
import sys
import struct
import numpy as np
import os, cv2, zipfile, io, re, glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from PIL import Image
from keras.models import model_from_json
from keras.models import Model, load_model
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Data aquisition
# ZIP read - Rewrite the path according to the location of the 'sg2t16_28000' folder.
z = zipfile.ZipFile('/sg2t16_28000/sgan_out.zip')
img_dirs = [ x for x in z.namelist() if re.search("^sgan_out/.*/$", x)]
print (img_dirs)
# Delete unnecessary strings
img_dirs = [ x.replace('sgan_out/', '') for x in img_dirs]
img_dirs = [ x.replace('/', '') for x in img_dirs]
img_dirs.sort()
print (img_dirs)
classes = img_dirs
num_classes = len(classes)
del img_dirs

# Image size
image_size = 256

#  Images acquisition array them
def im2array(path):
    X = []
    y = []
    class_num = 0

    for class_name in classes:
        if class_num == num_classes : break
        imgfiles = [ x for x in z.namelist() if re.search("^" + path + class_name + "/.*bmp$", x)] 
        for imgfile in imgfiles:
            # Image reading from ZIP
            image = Image.open(io.BytesIO(z.read(imgfile)))
            # RGB conversion
            image = image.convert('RGB')
            # Resize
            image = image.resize((image_size, image_size))
            # Convert from image to array
            data = np.asarray(image)
            X.append(data)
            y.append(classes.index(class_name))
        class_num += 1

    X = np.array(X)
    y = np.array(y)

    return X, y

# Test data acquisition
X_test, y_test = im2array("sgan_out/")
X_test = X_test.astype('float32')

# Normalization
X_test /= 255

# One-hot conversion
y_test = to_categorical(y_test, num_classes = num_classes)
print(y_test.shape)

# Load InceptionResNet V2 trained models and weights
model_dir = './model/'
weights_dir = './weights/'
f = open(model_dir +"BreastUS_original_crossInceptionResNetV2_775.json", 'r')
loaded_model_json = f.read()
f.close()
model = model_from_json(loaded_model_json)
model.load_weights(weights_dir + 'BreastUS_original_crossInceptionResNetV2_775.h5')

# Evaluation list
y_pred = model.predict(X_test, verbose=1)
y_pred_keras = model.predict(X_test, verbose=1)

# Test data n correct labels (n = 50000)
#true_classes = np.argmax(y_test[0:n], axis = 1)
true_classes = np.argmax(y_test[0:50000], axis = 1)
print('correct:', true_classes)

# Test data n correct labels (n = 50000)
pred_classes = np.argmax(model.predict(X_test[0:50000]), axis = 1)
print('prediction:', pred_classes)

# Create a copy destination folder
Malig_DIR = '/sg2t16_28000/select_malignancy_IRNV2/'
if not os.path.exists(Malig_DIR):
    os.mkdir(Malig_DIR)
Benign_DIR = '/sg2t16_28000/select_benign_IRNV2/'
if not os.path.exists(Benign_DIR):
    os.mkdir(Benign_DIR)

# Test data n images and prediction label / prediction probability output
for i in range(50000):
    if pred_classes[i] == 1 and true_classes[i] == 1:
        print('SelectedMalig' , str(i))
        plt.imsave(Malig_DIR + 'SelectedMalig' + str(i) +'.bmp', X_test[i])
    elif pred_classes[i] == 0 and true_classes[i] == 0:
        print('SelectedBenign' , str(i))
        plt.imsave(Benign_DIR + 'SelectedBenign' + str(i) +'.bmp', X_test[i])
    else:
        pass


