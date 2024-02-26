import cv2
import os
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from matplotlib.colors import ListedColormap
from skimage import measure
from matplotlib import cm
import numpy as np
import io
def get_matching_true_ids(true_label, pred_label):

    true_ids, pred_ids = [], []

    for pred_cell in np.unique(pred_label[pred_label > 0]):
        pred_mask = pred_label == pred_cell
        overlap_ids, overlap_counts = np.unique(true_label[pred_mask], return_counts=True)

        # get ID of the true cell that overlaps with pred cell most
        true_id = overlap_ids[np.argmax(overlap_counts)]

        true_ids.append(true_id)
        pred_ids.append(pred_cell)

    return true_ids, pred_ids

def get_cell_size(label_list, label_map):
    size_list = []
    for label in label_list:
        size = np.sum(label_map == label)
        size_list.append(size)

    return size_list

def label_image_by_ratio(true_label, pred_label, threshold=2):

    true_ids, pred_ids = get_matching_true_ids(true_label, pred_label)

    true_sizes = get_cell_size(true_ids, true_label)
    pred_sizes = get_cell_size(pred_ids, pred_label)
    fill_val = -threshold + 0.02
    disp_img = np.full_like(pred_label.astype('float32'), fill_val)
    for i in range(len(pred_ids)):
        current_id = pred_ids[i]
        true_id = true_ids[i]
        if true_id == 0:
            ratio = threshold
        else:
            ratio = np.log2(pred_sizes[i] / true_sizes[i])
        mask = pred_label == current_id
        boundaries = find_boundaries(mask, mode='inner')
        mask[boundaries > 0] = 0
        if ratio > threshold:
            ratio = threshold
        if ratio < -threshold:
            ratio = -threshold
        disp_img[mask] = ratio

    disp_img[-1, -1] = -threshold
    disp_img[-1, -2] = threshold

    return disp_img

def apply_colormap_to_img(label_img):
    coolwarm = cm.get_cmap('coolwarm', 256)
    newcolors = coolwarm(np.linspace(0, 1, 256))
    black = np.array([0, 0, 0, 1])
    newcolors[1:2, :] = black
    newcmp = ListedColormap(newcolors)

    transformed = np.copy(label_img)
    transformed -= np.min(transformed)
    transformed /= np.max(transformed)

    transformed = newcmp(transformed)

    return transformed

def log2_image_mask(truth_folder,pre_folder,save_folder):
    # 读取原图
    for truth_name in os.listdir(truth_folder):
        truth_path = os.path.join(truth_folder,truth_name)
        pre_name=truth_name
        pre_path = os.path.join(pre_folder,pre_name)
        truth_mask = cv2.imread(truth_path,0)
        pre_labels = cv2.imread(pre_path,0)
        truth_labels = truth_mask
        truth_labels = measure.label(truth_labels,connectivity=1)
        pre_labels[pre_labels>1]=1
        pre_labels = measure.label(pre_labels,connectivity=1)
        disp_img_mesmer = label_image_by_ratio(truth_labels, pre_labels)
        disp_img_mesmer_final = apply_colormap_to_img(disp_img_mesmer)
        plt.imsave('{}/{}'.format(save_folder,pre_name),disp_img_mesmer_final)
truth_folder='./dataset/test/masks'
pre_folder = './U-Net-patch_pre'
save_folder='./log2_fig'
log2_image_mask(truth_folder,pre_folder,save_folder)
# fig, ax = plt.subplots()
# pos = ax.imshow(disp_img_mesmer_final, cmap='coolwarm')
# fig.colorbar(pos)
# plt.savefig(os.path.join(data_dir, 'Figure_2e_colorbar.pdf'))
