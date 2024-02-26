from __future__ import print_function
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
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from tensorflow.python.keras.models import Model
from keras import backend as K
import utils.recompose as rp

# **Parameters for Image Resolution**

# In[ ]:


im_width = 256
im_height = 256


# **Function to load Images and Masks**

# In[ ]:

# Function to load Images and Masks
def get_images(parent_dir):
    tissue_dir = parent_dir
    im_names = os.listdir(tissue_dir)
    X = np.zeros((len(im_names), 10240, 10240, 3), dtype=np.float32)
    # y = np.zeros((len(im_names), im_height, im_width, 1), dtype=np.float32)
    id = 1
    for im_name in im_names:
    # Load images
        img = im_name
        img = load_img(tissue_dir+img)
        x_img = img_to_array(img)
        x_img = resize(x_img, (10240,10240, 3), mode = 'constant', preserve_range = True)
        # Load masks
		# mask_id_ = id_.split('.')[0]+'.png'
		# # mask_id_ = '-'.join(id_.split('-')[:-1]) + '_bin_mask-'+ id_.split('-')[-1]
		# # if gt_extension:
		# # 	mask_id_ = mask_id_.split('.')[0] + '.' + gt_extension
		# mask = img_to_array(load_img(gt_dir+mask_id_, grayscale=True))
		# mask = resize(mask, (im_width, im_height, 1), mode = 'constant', preserve_range = True)
        X[id-1] = x_img/255.0
        id += 1
        # y[n] = mask/255.0
    return X


# In[4]:


# Load the training images and mask
X_test = get_images("./test/")
patches_imgs_test, extended_height, extended_width = rp.get_data_testing_overlap(
        X_test,  # original
        n_test_images=100,
        patch_height=256,
        patch_width=256,
        stride_height=128,
        stride_width=128,
        channel=3)


# **Define the functions of Dice, Precision, Recall and F1**

# In[ ]:


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


# **Compile the Model**

# In[10]:


from Models import UNET
input_img = Input((im_height, im_width, 3), name='img')
model = UNET.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", f1_m, precision_m, recall_m, dice_coef])


# **Load the model weights**

# In[ ]:


model.load_weights('./model-unet.h5')


# **Evaluate on test dataset set and print the results**

# In[13]:


# loss, accuracy, f1_score, precision, recall , dice_score = model.evaluate(X_test, y_test, verbose=1)
# print("unet_loss:", loss)
# print("unet_Accuracy:", accuracy)
# print("unet_f1_score:", f1_score)
# print("unet_dice_score:", dice_score)


# **Predict on train and test**

# In[14]:

patches_imgs_test = np.transpose(patches_imgs_test,(0,2,3,1))
preds_test = model.predict(patches_imgs_test, verbose=1)
use_weight = 1
loss_weight = rp.get_loss_weight(im_height, im_width, use_weight,border=16)
preds_test = (preds_test > 0.5).astype(np.uint8)
preds_test = np.transpose(preds_test,(0,3,1,2))
pred_img = rp.recompose_overlap(preds_test, 10240,10240, 128, 128,1, loss_weight=loss_weight)
pred_img = pred_img[:, :, 0:10240, 0:10240]
test_result = '/home/ranran/desktop/UNET/pre/'

for i in range(pred_img.shape[0]):
    rp.visualize(np.transpose(pred_img[i,:,:,:], (1,2,0)), test_result + str(i+1) )

# **Threshold predictions**

# In[ ]:
#
#
# preds_test_t = (preds_test > 0.5).astype(np.uint8)
#
#
# # **Threshold predictions**
#
# # In[ ]:
#
#
# def plot_sample(X, y, preds, binary_preds, ix=None):
#     """Function to plot the results"""
#     if ix is None:
#         ix = random.randint(0, len(X)-1)
#     has_mask = y[ix].max() > 0
#     fig, ax = plt.subplots(1, 4, figsize=(20, 10))
#     ax[0].imshow(X[ix])
#     if has_mask:
#         ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#     ax[0].set_title('Image')
#     ax[1].imshow(y[ix].squeeze())
#     ax[1].set_title('Ground Truth')
#     ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
#     if has_mask:
#         ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#     ax[2].set_title('Predicted')
#     ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
#     if has_mask:
#         ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#     ax[3].set_title('Predicted binary')
#     plt.show()
#
# # **Testing Images With Ground Truth and Predicted**
#
# # In[17]:
#
#
# plot_sample(X_test, y_test, preds_test, preds_test_t, ix=5)

# plot_sample(X_test, y_test, preds_test, preds_test_t)