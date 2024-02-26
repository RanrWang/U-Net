'''environment: NSE_tf'''

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# **Parameters for Image Resolution**

# In[ ]:


im_width = 256
im_height = 256


# **Function to load Images and Masks**

# In[ ]:


def get_images(parent_dir, im_shape, img_folder="imgs/", gt_folder="masks/", gt_extension=None):
    tissue_dir = parent_dir + img_folder
    gt_dir = parent_dir + gt_folder
    im_width, im_height = im_shape
    ids = next(os.walk(tissue_dir))[2]
    print("No. of images = ", len(ids))
    X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    # tqdm is used to display the progress bar
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        img = load_img(tissue_dir + id_)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_width, im_height, 3), mode='constant', preserve_range=True)
        # Load masks
        mask_id_ = id_.split('.')[0] + '.png'
        # mask_id_ = '-'.join(id_.split('-')[:-1]) +'_bin_mask-'+ id_.split('-')[-1]
        # if gt_extension:
        # 	mask_id_ = mask_id_.split('.')[0] + '.' + gt_extension
        mask = img_to_array(load_img(gt_dir+mask_id_, grayscale=True))
        # mask = resize(mask, (im_width, im_height, 1), mode = 'constant', preserve_range = True)
        # mask = np.load(gt_dir + mask_id_, allow_pickle=True)
        mask = resize(mask, (im_width, im_height, 1), mode='constant', preserve_range=True)
        X[n] = x_img / 255.0
        y[n] = mask / 255.0

    return X, y


# **Load the training images and mask**

# In[ ]:


X, y = get_images("./ConSep_dataset/train/", (im_width, im_height),gt_extension = 'png')
# X, y = get_images("./data/dataset/train/", (im_width, im_height))

# **Split the training data into training and validation**

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


# **Visualize any random image along with the mask**


ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() >0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 15))
ax1.imshow(X_train[ix, ..., 0], cmap = 'seismic', interpolation = 'bilinear')
if has_mask:
    ax1.contour(y_train[ix].squeeze(), colors = 'k', linewidths = 5, levels = [0.5])
ax1.set_title('TissueImage')
ax2.imshow(y_train[ix].squeeze(), cmap = 'gray', interpolation = 'bilinear')
ax2.set_title('GroundTruth')

from Models import UNET

# **Define the functions of Dice, Precision, Recall and F1**

# In[ ]:


from keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


binary_crossentropy = "binary_crossentropy"


# **Set Parameters and compile the model**

# In[ ]:


input_img = Input((im_height, im_width, 3), name='img')
model = UNET.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=False)
model.compile(optimizer=RMSprop(), loss=[binary_crossentropy], metrics=["accuracy", f1_m, precision_m, recall_m, dice_coef])


# **To save model, Early Stopping and Reduce Learning rate use callbacks**

# In[ ]:


callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
    ModelCheckpoint('model-unet.h5', save_best_only=False, save_weights_only=True, verbose=1)
]


# **Model Summary**

# In[ ]:


model.summary()


# **Train the model**

# In[ ]:


results = model.fit(X_train, y_train, batch_size=32, 
                    epochs=50, callbacks=callbacks, validation_data=(X_valid, y_valid), verbose = 1)


# **Graph between Training and Validation Loss**

# In[ ]:


plt.figure(figsize=(8, 8))
plt.title("Learning curve - Loss")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();


# **Graph between Training and Validation Accuracy**

# In[ ]:


plt.figure(figsize=(8, 8))
plt.title("Learning curve - Accuracy")
plt.plot(results.history["acc"], label="accuracy")
plt.plot(results.history["val_acc"], label="val_accuracy")
plt.plot( np.argmax(results.history["val_acc"]), np.max(results.history["val_acc"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend();


# **Graph between Training and Validation Dice**

# In[ ]:


plt.figure(figsize=(8, 8))
plt.title("Learning curve - Dice Coefficient")
plt.plot(results.history["dice_coef"], label="dice_coef")
plt.plot(results.history["val_dice_coef"], label="val_dice_coef")
plt.plot( np.argmax(results.history["val_dice_coef"]), np.max(results.history["val_dice_coef"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Dice Coefficient")
plt.legend();


# **Graph between Training and Validation F1 score**

# In[ ]:


plt.figure(figsize=(8, 8))
plt.title("Learning curve - F1 Score")
plt.plot(results.history["f1_m"], label="F1 Score")
plt.plot(results.history["val_f1_m"], label="val F1 Score")
plt.plot( np.argmax(results.history["val_f1_m"]), np.max(results.history["val_f1_m"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.legend();

