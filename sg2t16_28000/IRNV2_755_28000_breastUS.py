import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os, cv2, zipfile, io, re, glob
from PIL import Image
from sklearn.model_selection import KFold
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Activation, Dropout
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# Data acquisition
# ZIP read - Rewrite the path according to the location of the 'sg2t16_28000' folder.
z = zipfile.ZipFile('/sg2t16_28000/hikakudata.zip')
img_dirs = [ x for x in z.namelist() if re.search("^hikakudata/Train/.*/$", x)]

# Delete unnecessary strings
img_dirs = [ x.replace('hikakudata/Train/', '') for x in img_dirs]
img_dirs = [ x.replace('/', '') for x in img_dirs]
img_dirs.sort()
classes = img_dirs
num_classes = len(classes)
del img_dirs

# Image size
image_size = 256

# Images acquisition array them
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
X_train, y_train = im2array("hikakudata/Train/")

# Test data acquisition
X_train = X_train.astype('float32')

# Normalization
X_train /= 255

# one-hot conversion
y_train = to_categorical(y_train, num_classes = num_classes)

# Split valid data from train data klearn.model_selection import Use KFold (10 splits)
kf = KFold(n_splits=10, shuffle=True)

for train_index, val_index in kf.split(X_train,y_train):

    X_tra=X_train[train_index]
    y_tra=y_train[train_index]
    X_val=X_train[val_index]
    y_val=y_train[val_index]

# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False
)

# EarlyStopping
early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    verbose = 1
)

# ModelCheckpoint
weights_dir = './weights/'
if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    period = 3
)

# Reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 3,
    verbose = 1
)

# Log for TensorBoard
logging = TensorBoard(log_dir = "log/")

# Model learning
def model_fit():
    hist = model.fit_generator(
        datagen.flow(X_train, y_train, batch_size = 32),
        steps_per_epoch = X_train.shape[0] // 32,
        epochs = 50,
        validation_data = (X_val, y_val),
        callbacks = [early_stopping, reduce_lr],
        shuffle = True,
        verbose = 1
    )
    return hist

# Save result text
nb_epoch = 50

result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def save_history(hist, result_file):
    loss_v = hist.history['loss']
    acc_v = hist.history['acc']
    val_loss_v = hist.history['val_loss']
    val_acc_v = hist.history['val_acc']
    nb_epoch = len(acc_v)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss_v[i], acc_v[i], val_loss_v[i], val_acc_v[i]))

# Save the model
model_dir = './model/'
weights_dir = './weights/'
if os.path.exists(model_dir) == False : os.mkdir(model_dir)

def model_save(model_name):
    model_json = model.to_json()
    with open(model_dir +"BreastUS_SG2T16_28000_cross" + model_name + ".json", "w") as json_file:
     json_file.write(model_json)
    model.save(model_dir + 'model_' + model_name + '.hdf5')
    model.save_weights(weights_dir + 'BreastUS_SG2T16_28000_cross' + model_name + '.h5')
    # Save lightweight model without optimizer (cannot be trained or evaluated, but predictable)
    model.save(model_dir + 'model_' + model_name + '-opt.hdf5', include_optimizer = False)

# Plot the learning curve
def learning_plot(title):
    plt.figure(figsize = (18,6))

    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["acc"], label = "acc", marker = "o")
    plt.plot(hist.history["val_acc"], label = "val_acc", marker = "o")
    #plt.yticks(np.arange())
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = 'gray', alpha = 0.2)

    # loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label = "loss", marker = "o")
    plt.plot(hist.history["val_loss"], label = "val_loss", marker = "o")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = 'gray', alpha = 0.2)

    plt.show()

# Model evaluation
def model_evaluate():
    score = model.evaluate(X_val, y_val, verbose = 1)
    print("evaluate loss: {[0]:.4f}".format(score))
    print("evaluate acc: {[1]:.1%}".format(score))
    dt_now = datetime.datetime.now()
    fileobj = open("InceptionvResNet_V2_775_255_SG2T16_28000_results.txt", "w")

# Open sample.txt in export mode.
    fileobj.write('{0:%Y_%m_%d_%H_%M_%S}'.format(dt_now) + "evaluate loss: {[0]:.4f}".format(score) + ' ' + "evaluate acc: {[1]:.1%}".format(score))
    fileobj.close()

# InceptionResNetV2
base_model = InceptionResNetV2(
    include_top = False,
    weights = "imagenet",
    input_shape = None
)

# New construction of fully connected layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

# Network definition
model = Model(inputs = base_model.input, outputs = predictions)
print("{}層".format(len(model.layers)))

# Freeze up to 775 layers of network definition
for layer in model.layers[:775]:
    layer.trainable = False

    # Unfreeze Batch Normalization
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True

# Learn after 775 layers
for layer in model.layers[775:]:
    layer.trainable = True

# Compile after setting layer.trainable
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['acc']
)

hist=model_fit()
learning_plot("InceptionResNetV2_775_SG2T16_28000")
model_evaluate()
model_save("InceptionResNetV2_775_SG2T16_28000")
save_history(hist, os.path.join(result_dir, 'history_inceptionresnetv2_775_255_SG2T16_28000.txt'))

