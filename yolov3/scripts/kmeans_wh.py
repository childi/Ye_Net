import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


label_path = "../data/dataset/kitti_train3d.txt"
label_files = []
f = open(label_path)
for line in f:
    label_path = line.rstrip().replace('image', 'label')
    label_path = label_path.replace('mnt', 'media')
    label_path = label_path.replace('.png', '.txt')
    label_files.append(label_path)
    # label_files.append(line.rstrip())

f.close()

boxes = []
for label_file in label_files:
    f = open(label_file.split()[0])
    for line in f:
        temp = line.strip().split(" ")
        if len(temp) > 1 and temp[0] == str('Car'):
            # boxes.append([float(temp[6]) - float(temp[4]), float(temp[7]) - float(temp[5])])  # 2d wh
            boxes.append([float(temp[9]), float(temp[10])])  # 3d lw
        elif temp[0] == str('DontCare'):
            break

X = np.array(boxes)
# 绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker='.', label='see')
plt.xlabel('width')
plt.ylabel('length')
plt.legend(loc=2)
plt.show()

estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(X)  # 聚类
center = estimator.cluster_centers_
label_pred = estimator.labels_  # 获取聚类标签
loss = estimator.inertia_
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == 3]
x4 = X[label_pred == 4]
x5 = X[label_pred == 5]
x6 = X[label_pred == 6]
x7 = X[label_pred == 7]
x8 = X[label_pred == 8]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c="magenta", marker='x', label='label3')
plt.scatter(x4[:, 0], x4[:, 1], c="pink", marker='o', label='label4')
plt.scatter(x5[:, 0], x5[:, 1], c="orange", marker='^', label='label5')
plt.scatter(x6[:, 0], x6[:, 1], c="yellow", marker='.', label='label6')
plt.scatter(x7[:, 0], x7[:, 1], c="purple", marker=',', label='label7')
plt.scatter(x8[:, 0], x8[:, 1], c="cyan", marker='+', label='label8')
plt.scatter(center[:, 0], center[:, 1], c="black", marker='o', label='center')
plt.xlabel('width')
plt.ylabel('length')
plt.legend(loc=2)
plt.show()

# print result
print("k-means result：\n")
for centroid in center:
    print(centroid[0], centroid[1])
print(loss)
