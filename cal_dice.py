import cv2
import os
import numpy as np
from skimage import io, transform
from skimage import measure
from utils import stats_utils

def cal_stats(im_data_folder, gt_folder):
    dice2 = []
    aji = []
    dq = []
    sq = []
    pq = []
    dicet = []
    IOU_list=[]
    for im_name in os.listdir(im_data_folder):

        im_path=os.path.join(im_data_folder,im_name)
        # gt_name = im_name.replace('.png','.npy')
        gt_path=os.path.join(gt_folder,im_name)
        s2 = cv2.imread(im_path, 0)  # 模板
        # s1 = cv2.imread(gt_path, 0)
        s1 = cv2.imread(gt_path,0)
        if len(np.unique(s1)) == 1 or len(np.unique(s2)) == 1:
            pq_ = np.nan
            aji_ = np.nan
            dice_t = np.nan
            dice2_ = np.nan
            dq_ = np.nan
            sq_ = np.nan
            IOU = np.nan
        else:
            dice_t=stats_utils.get_dice_1(s1,s2)
            aji_ = stats_utils.get_fast_aji(s1, s2)
            dice2_=stats_utils.get_fast_dice_2(s1,s2)
            IOU=stats_utils.iou_metric(s1,s2)
            qq,array = stats_utils.get_fast_pq(s1,s2)
            dq_ = qq[0]
            sq_ = qq[1]
            pq_ = qq[2]
        dicet.append(dice_t)
        aji.append(aji_)
        dice2.append(dice2_)
        IOU_list.append(IOU)
        dq.append(dq_)
        sq.append(sq_)
        pq.append(pq_)
    return dicet,dice2,IOU_list,aji,dq,sq,pq
prec='./patch-pre'
gt='./ConSep_dataset/test/masks'
all_dice,all_dice2,all_IOU,all_aji,all_dq,all_sq,all_pq = cal_stats(prec,gt)
mean_dice = np.nanmean(all_dice)
mean_dice2 = np.nanmean(all_dice2)
mean_aji = np.nanmean(all_aji)
mean_IOU = np.nanmean(all_IOU)
mean_dq = np.nanmean(all_dq)
mean_sq = np.nanmean(all_sq)
mean_pq = np.nanmean(all_pq)
# print("图的dice为{}".format(all_dice))
print("图的mean_dice为{}".format(mean_dice))
print("图的mean_dice2为{}".format(mean_dice2))
print("图的mean_IOU为{}".format(mean_IOU))
# print("图的aji为{}".format(all_aji))
print("图的mean_aji为{}".format(mean_aji))
# print("图的pq为{}".format(all_dq))
print("图的mean_dq为{}".format(mean_dq))
# print("图的pq为{}".format(all_sq))
print("图的mean_sq为{}".format(mean_sq))
# print("图的pq为{}".format(all_pq))
print("图的mean_pq为{}".format(mean_pq))

