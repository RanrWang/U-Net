import numpy as np
from skimage import measure,color,segmentation,feature
from skimage.morphology import remove_small_objects,remove_small_holes
import matplotlib.pyplot as plt
import cv2,os
from skimage import data
from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy import ndimage
import random
def de_convolution(image_folder,save_folder):
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)  # 标注文件路径
        # Create an artificial color close to the orginal one
        cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
        cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
                                                     'saddlebrown'])
        cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',
                                                       'white'])
        # ihc_rgb = data.immunohistochemistry()
        ihc_rgb = cv2.imread(image_path)
        ihc_rgb = cv2.cvtColor(ihc_rgb,cv2.COLOR_BGR2RGB)
        ihc_hed = rgb2hed(ihc_rgb)
        h_image = ihc_hed[:,:,0]
        e_image = ihc_hed[:,:,1]
        dab_image = ihc_hed[:,:,2]
        plt.imsave('{}/{}'.format(save_folder,image_name), dab_image,cmap=cmap_dab)
        # path = save_folder + '/' + image_name
        # return path
im_data_folder = './patch'
mask_folder = './pre_patch'
save_folder = './dab'
# de_convolution(im_data_folder,save_folder)
dab_folder = './dab'
for im_name in os.listdir(im_data_folder):
    im_path=os.path.join(im_data_folder,im_name)
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mask_path = os.path.join(mask_folder,im_name)
    dab_path = os.path.join(dab_folder,im_name)
    # path = de_convolution(image_folder,save_folder)
    mask = cv2.imread(mask_path)
    dab_img = cv2.imread(dab_path)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask[mask>1]=1
    dab_img = cv2.cvtColor(dab_img,cv2.COLOR_BGR2RGB)
    threshold = cv2.cvtColor(dab_img,cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(threshold, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    threshold[threshold>1]=1
    threshold=remove_small_objects(np.array(threshold, dtype= bool), min_size=80)
    threshold=np.array(threshold).astype(np.uint8)
    mask=remove_small_objects(np.array(mask, dtype= bool), min_size=80)
    mask=np.array(mask).astype(np.uint8)
    labels_image=measure.label(mask,connectivity=2)
    mask_list = list(np.unique(labels_image))
    mask_list.remove(0)
    out = np.zeros(( labels_image.shape[0], labels_image.shape[1]))
    for i in mask_list:
        region = labels_image == i
        region = np.array(region).astype(np.uint8)
        region_size = np.sum(region == 1)
        th = threshold[np.where(region==1)]
        positive_region = np.sum(th==1)

        if positive_region  >= region_size * 0.3:
            mask[np.where(region==1)] = 0
            out[np.where(region==1)] = 1
        else:
            pass
    out=np.array(out).astype(np.uint8)
    contours, _ = cv2.findContours(
                mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
            )
    image = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    contours2, _ = cv2.findContours(
                out, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
            )
    image = cv2.drawContours(image, contours2, -1, (255, 0, 0), 2)
    plt.imsave('./overlay_save/{}'.format(im_name),image)

