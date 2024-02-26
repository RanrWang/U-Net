#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input

# In[2]:


# 设置图像大小
im_width = 256
im_height = 256


# In[3]:


# Function to load Images and Masks
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

    return X, y,ids


# In[4]:


# Load the training images and mask
X_test, y_test,ids = get_images("./ConSep_dataset/test/", (im_width, im_height))

# In[5]:


# Define the DeepLabV3plus Model Function

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
model = UNET.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=False)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", f1_m, precision_m, recall_m, dice_coef])


# **Load the model weights**

# In[ ]:


model.load_weights('./model-unet.h5')


# In[10]:

from PIL import Image
# Threshold predictions
preds_test_t = model.predict(X_test, verbose=1)
# preds_test_t = (preds_test_t > 0).astype(np.uint8)
for i in range(len(X_test)):
    id = ids[i]
    preds = preds_test_t[i,:,:,0]
    preds = (preds > 0.5).astype(np.uint8)
    img = Image.fromarray((preds * 255).astype(np.uint8))  # the image is between 0-1
    filename = './patch-pre/' + id
    img.save(filename)


# preds_test_t = (preds_test > 0.5).astype(np.uint8)


# In[11]:
#
#
# # Function to plot the Images, Ground Trtuh and Predicted
# def plot_sample(X, y, preds, binary_preds, ix=None):
#     """Function to plot the results"""
#     if ix is None:
#         ix = random.randint(0, len(X) - 1)
#     has_mask = y[ix].max() > 0
#     fig, ax = plt.subplots(1, 4, figsize=(20, 10))
#     ax[0].imshow(X[ix])
#     if has_mask:
#         ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#     ax[0].set_title('Image')
#     ax[1].imshow(y[ix].squeeze(),cmap=plt.cm.jet)
#     ax[1].set_title('Ground Truth')
#     ax[2].imshow(preds[ix].squeeze(),cmap=plt.cm.jet)
#     if has_mask:
#         ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#     ax[2].set_title('Predicted')
#     ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
#     if has_mask:
#         ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
#     ax[3].set_title('Predicted binary')
#
#
# # In[28]:
#
#
# # Testing Images With Ground Truth and Predicted
# plot_sample(X_test, y_test, preds_test, preds_test, ix=5)
#
# # In[14]:
#
#
# plot_sample(X_test, y_test, preds_test, preds_test)
#
# # In[26]:
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# img1 = X_test[5]
# img1 = np.squeeze(img1)
# img2 = preds_test[5]
# # np.save('./a.npy',img2)
# img2 = np.squeeze(img2)
# plt.figure()
# fig = plt.figure(dpi=1000)
# plt.subplot(1, 2, 1)
# plt.imshow(img1)
# plt.title("Tissue Image")
# plt.subplot(1, 2, 2)
# plt.imshow(img2, 'jet', interpolation='bilinear', alpha=0.9)
# plt.title("Superimposed with Predicted")
# plt.show()
# plt.savefig("plot.png", format="png")
#
# # In[21]:
#
#
# import warnings
#
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy
# from scipy.optimize import linear_sum_assignment
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
#
#
# def get_fast_aji(true, pred):
#     """
#     AJI version distributed by MoNuSeg, has no permutation problem but suffered from
#     over-penalisation similar to DICE2
#     Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
#     not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
#     effect on the result.
#     """
#     true = np.copy(true)  # ? do we need this
#     pred = np.copy(pred)
#     true_id_list = list(np.unique(true))
#     pred_id_list = list(np.unique(pred))
#
#     true_masks = [None, ]
#     for t in true_id_list[1:]:
#         t_mask = np.array(true == t, np.uint8)
#         true_masks.append(t_mask)
#
#     pred_masks = [None, ]
#     for p in pred_id_list[1:]:
#         p_mask = np.array(pred == p, np.uint8)
#         pred_masks.append(p_mask)
#
#     # prefill with value
#     pairwise_inter = np.zeros([len(true_id_list) - 1,
#                                len(pred_id_list) - 1], dtype=np.float64)
#     pairwise_union = np.zeros([len(true_id_list) - 1,
#                                len(pred_id_list) - 1], dtype=np.float64)
#
#     # caching pairwise
#     for true_id in true_id_list[1:]:  # 0-th is background
#         t_mask = true_masks[true_id]
#         pred_true_overlap = pred[t_mask > 0]
#         pred_true_overlap_id = np.unique(pred_true_overlap)
#         pred_true_overlap_id = list(pred_true_overlap_id)
#         for pred_id in pred_true_overlap_id:
#             if pred_id == 0:  # ignore
#                 continue  # overlaping background
#             p_mask = pred_masks[pred_id]
#             total = (t_mask + p_mask).sum()
#             inter = (t_mask * p_mask).sum()
#             pairwise_inter[true_id - 1, pred_id - 1] = inter
#             pairwise_union[true_id - 1, pred_id - 1] = total - inter
#     #
#     pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
#     # pair of pred that give highest iou for each true, dont care
#     # about reusing pred instance multiple times
#     paired_pred = np.argmax(pairwise_iou, axis=1)
#     pairwise_iou = np.max(pairwise_iou, axis=1)
#     # exlude those dont have intersection
#     paired_true = np.nonzero(pairwise_iou > 0.0)[0]
#     paired_pred = paired_pred[paired_true]
#     # print(paired_true.shape, paired_pred.shape)
#     overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
#     overall_union = (pairwise_union[paired_true, paired_pred]).sum()
#     #
#     paired_true = (list(paired_true + 1))  # index to instance ID
#     paired_pred = (list(paired_pred + 1))
#     # add all unpaired GT and Prediction into the union
#     unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
#     unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
#     for true_id in unpaired_true:
#         overall_union += true_masks[true_id].sum()
#     for pred_id in unpaired_pred:
#         overall_union += pred_masks[pred_id].sum()
#     #
#     aji_score = overall_inter / overall_union
#     return aji_score
#
#
# # In[22]:
#
#
# def remap_label(pred, by_size=False):
#     """
#     Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
#     not [0, 2, 4, 6]. The ordering of instances (which one comes first)
#     is preserved unless by_size=True, then the instances will be reordered
#     so that bigger nucler has smaller ID
#     Args:
#         pred    : the 2d array contain instances where each instances is marked
#                   by non-zero integer
#         by_size : renaming with larger nuclei has smaller id (on-top)
#     """
#     pred_id = list(np.unique(pred))
#     pred_id.remove(0)
#     if len(pred_id) == 0:
#         return pred  # no label
#     if by_size:
#         pred_size = []
#         for inst_id in pred_id:
#             size = (pred == inst_id).sum()
#             pred_size.append(size)
#         # sort the id by size in descending order
#         pair_list = zip(pred_id, pred_size)
#         pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
#         pred_id, pred_size = zip(*pair_list)
#
#     new_pred = np.zeros(pred.shape, np.int32)
#     for idx, inst_id in enumerate(pred_id):
#         new_pred[pred == inst_id] = idx + 1
#     return new_pred
#
#
# # In[27]:
#
#
# print(get_fast_aji(remap_label(np.squeeze(y_test[5])), remap_label(np.squeeze(preds_test[5]))))
#
# # In[ ]:



