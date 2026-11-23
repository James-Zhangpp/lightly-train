import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt


# 指定单张图像路径
mask_path = r'.\Breakage_small_seg_1024x416\ann_dir\train\aoi_ng_011.png'
img_path = r'.\Breakage_small_seg_1024x416\img_dir\train\aoi_ng_011.png'

img = cv2.imread(img_path)
mask = cv2.imread(mask_path)
print(img.shape)
print(mask.shape)

# mask 语义分割标注，与原图大小相同
print(np.unique(mask))


plt.figure(figsize=(10, 6))
plt.imshow(mask*50)
plt.axis('off')
plt.show()


# 每个类别的 BGR 配色
palette = [
    ['FPC', [127,127,127]]
]

palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]

mask = mask[:, :, 0]

# 将整数ID，映射为对应类别的颜色
viz_mask_bgr = np.zeros((mask.shape[0], mask.shape[1], 3))
for idx in palette_dict.keys():
    viz_mask_bgr[np.where(mask == idx)] = palette_dict[idx]
viz_mask_bgr = viz_mask_bgr.astype('uint8')

# 将语义分割标注图和原图叠加显示
opacity = 0.2  # 透明度越大，可视化效果越接近原图
label_viz = cv2.addWeighted(img, opacity, viz_mask_bgr, 1 - opacity, 0)
# %%
plt.figure(figsize=(10, 6))
plt.imshow(label_viz[:, :, ::-1])
plt.axis('off')
plt.show()


cv2.imwrite('outputs/D-1.jpg', label_viz)