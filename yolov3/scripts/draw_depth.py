import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
# image = np.load("/media/dataset/Kitti/object/training/dense_map_only_lidar/000025.npy")
image = np.load("/media/personal_data/zhangye/bev_input/6.npy")
# plt.imshow(image[:, :])
# plt.show()

# for i in range(0, image.shape[-1]):
#     plt.imshow(image[:, :, i])
#     cv2.imwrite(str(i)+".png", image[:, :, i])
#     plt.show()

# fig = plt.figure()  # 获取到当前figure对象
# ax = fig.gca(projection='3d')
# ax.scatter(image[0, :], image[1, :], image[2, :], c='b', s=1)  #
# plt.show()

ax = plt.subplot()
ax.scatter(image[0, :], image[1, :], c='k', s=0.5)
plt.show()

