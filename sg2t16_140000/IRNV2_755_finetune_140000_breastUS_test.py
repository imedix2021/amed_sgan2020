from __future__ import print_function
import sys
print(sys.version)
print(sys.path)
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
# ZIP read - Rewrite the path according to the location of the 'sg2t16_140000' folder.
z = zipfile.ZipFile('/sg2t16_140000/hikakudata.zip')
img_dirs = [ x for x in z.namelist() if re.search("^hikakudata/Test/.*/$", x)]
print (img_dirs)
# Delete unnecessary strings
img_dirs = [ x.replace('hikakudata/Test/', '') for x in img_dirs]
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

# Train data acquisition
X_test, y_test = im2array("hikakudata/Test/")

# Test data acquisition
X_test = X_test.astype('float32')

# Normalization
X_test /= 255

# One-hot conversion
y_test = to_categorical(y_test, num_classes = num_classes)
print(y_test.shape)

# Load InceptionResNet V2 trained models and weights
model_dir = './model/'
weights_dir = './weights/'
f = open(model_dir +"BreastUS_SG2T16_140000_crossInceptionResNetV2_775.json", 'r')
loaded_model_json = f.read()
f.close()
model = model_from_json(loaded_model_json)
model.load_weights(weights_dir + 'BreastUS_SG2T16_140000_crossInceptionResNetV2_775.h5')

# Evaluation list
y_pred = model.predict(X_test, verbose=1)
y_pred_keras = model.predict(X_test, verbose=1)

# Test data n correct labels (n = 300)
true_classes = np.argmax(y_test[0:300], axis = 1)
print('correct:', true_classes)
np.savetxt('correctIncResNetV2_775_140000.csv',true_classes,delimiter=',')

# Test data n predictive labels (n = 300)
pred_classes = np.argmax(model.predict(X_test[0:300]), axis = 1)
print('prediction:', pred_classes)
np.savetxt('predictionsIncResNetV2_775_140000.csv',pred_classes,delimiter=',')

# Confusion Matrix
print('2x2_IncRenNetV2_775_sg2t16_140000')
# Benign Prediction
#                     Predicted
#                 Malig(0)  Benign(1)
# Actural Malig(0)    TN       FP
#        Benign(1)    FN       TP
print("Benign")
print(confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1), labels=[1,0]))
conf_matrix_benign = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1), labels=[1,0])
TP_benign = conf_matrix_benign[1][1]
TN_benign = conf_matrix_benign[0][0]
FP_benign = conf_matrix_benign[0][1]
FN_benign = conf_matrix_benign[1][0]

# Calculate precision
conf_precision_benign = (TP_benign / float(TP_benign + FP_benign))
print ("Benign precision:", conf_precision_benign, "(", TP_benign, "/", float(TP_benign + FP_benign), ")")
# Calculate the sensitivity(recall)
conf_sensitivity_benign = (TP_benign / float(TP_benign + FN_benign))
print ("Benign recall(sensitivity):", conf_sensitivity_benign, "(", TP_benign, "/", float(TP_benign + FN_benign), ")")
# Calculate the specificity
conf_specificity_benign = (TN_benign / float(TN_benign + FP_benign))
print ("Benign specificity:", conf_specificity_benign, "(", TN_benign, "/", float(TN_benign + FP_benign), ")")
# Calculate f_1 score
conf_f1_benign = 2 * ((conf_precision_benign * conf_sensitivity_benign) / (conf_precision_benign + conf_sensitivity_benign))
print ("Benign f_1 score:", conf_f1_benign, "(", 2 * (conf_precision_benign * conf_sensitivity_benign), "/", conf_precision_benign + conf_sensitivity_benign, ")")

# Malig Prediction
#                     Predicted
#                  Benign(0)  Malig(1)
# Actural Benign(0)    TN      FP
#         Benign(1)    FN       TP
print()
print("Malignancy")
print(confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1)))
conf_matrix_malig = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
TP_malig = conf_matrix_malig[1][1]
TN_malig = conf_matrix_malig[0][0]
FP_malig = conf_matrix_malig[0][1]
FN_malig = conf_matrix_malig[1][0]

# Calculate precision
conf_precision_malig = (TP_malig / float(TP_malig + FP_malig))
print ("Malig precision:", conf_precision_malig, "(", TP_malig, "/", float(TP_malig + FP_malig), ")")
# Calculate the sensitivity(recall)
conf_sensitivity_malig = (TP_malig / float(TP_malig + FN_malig))
print ("Malig recall(sensitivity):", conf_sensitivity_malig, "(", TP_malig, "/", float(TP_malig + FN_malig), ")")
# Calculate the specificity
conf_specificity_malig = (TN_malig / float(TN_malig + FP_malig))
print ("Malig specificity:", conf_specificity_malig, "(", TN_malig, "/", float(TN_malig + FP_malig), ")")
# Calculate f_1 score
conf_f1_malig = 2 * ((conf_precision_malig * conf_sensitivity_malig) / (conf_precision_malig + conf_sensitivity_malig))
print ("Malig f_1 score:", conf_f1_malig, "(", 2 * (conf_precision_malig * conf_sensitivity_malig), "/", conf_precision_malig + conf_sensitivity_malig, ")")
print()
# Calculate accuracy
conf_accuracy = (float (TP_malig+TN_malig) / float(TP_malig + TN_malig + FP_malig + FN_malig))
print ("Accuracy:", conf_accuracy, "(", TP_malig+TN_malig, "/",float(TP_malig + TN_malig + FP_malig + FN_malig), ")")

print()
# classification_report by sklearn
# 0 = Benign, 1 = Malig
print(classification_report(np.argmax(y_test, 1), np.argmax(y_pred, 1), digits=3, target_names=['Benign', 'Malignant']))

# ROC Curve
y_pred_keras = model.predict(X_test, verbose=1)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test[:,0], y_pred_keras[:,0])
plt.plot(fpr_keras, tpr_keras, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.savefig('/sg2t16_140000/Real_BreastUS_IRNV2_Test_roc_curve.png')

# AUC
auc_keras = auc(fpr_keras, tpr_keras)
print('AUC:', auc_keras)



