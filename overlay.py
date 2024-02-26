import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def union_image_mask(image_folder, mask_folder,save_folder):
    # 读取原图
    for im_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder,im_name)
        mask_name = im_name.replace('.png','.png')
        mask_path = os.path.join(mask_folder,mask_name)
        save_path = os.path.join(save_folder,mask_name)
        image = cv2.imread(image_path)
        # 读取分割mask，这里本数据集中是白色背景黑色mask
        mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_2d= (mask_2d>0.5)
        inst_map = mask_2d
        inst_map = inst_map.astype(np.uint8)
        contours, _ = cv2.findContours(
            inst_map, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
        )
        image = cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        # # 打开画了轮廓之后的图像
        # plt.imshow(image)
        # plt.show()
        # 保存图像
        cv2.imwrite('{}'.format(save_path), image)
image_folder='./ConSep_dataset/test/imgs'
mask_folder = './patch-pre'
save_folder = './overlay'
union_image_mask(image_folder,mask_folder,save_folder)