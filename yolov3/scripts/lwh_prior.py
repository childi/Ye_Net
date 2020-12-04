import numpy as np
import os
import matplotlib.pyplot as plt

classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
h_list = [[] for _ in range(len(classes))]
w_list = [[] for _ in range(len(classes))]
l_list = [[] for _ in range(len(classes))]
z_list = [[] for _ in range(len(classes))]
with open('trainval.txt', 'r') as f:
    txt = f.readlines()
    image_inds = [line.strip() for line in txt]

for image_ind in image_inds:
    label_path = os.path.join('/media/dataset/Kitti/object/training/label_2/', image_ind + '.txt')
    with open(label_path, 'r') as la:
        txtl = la.readlines()
        label_file = [linel.strip() for linel in txtl]
    for label in label_file:
        label = label.split()
        if label[0] == 'DontCare':
            break
        class_ind = classes.index(label[0].strip())
        h = float(label[8])
        w = float(label[9])
        l = float(label[10])
        z = float(label[13])
        h_list[class_ind].append(h)
        w_list[class_ind].append(w)
        l_list[class_ind].append(l)
        z_list[class_ind].append(z)

for i in range(len(classes) - 1):
    h_mean = np.mean(h_list[i])
    h_median = h_list[i][round(len(h_list[i])/2)]
    print('%s class, h mean: %f  median: %f' % (classes[i], h_mean, h_median))
    w_mean = np.mean(w_list[i])
    w_median = w_list[i][round(len(w_list[i]) / 2)]
    print('%s class, w mean: %f  median: %f' % (classes[i], w_mean, w_median))
    l_mean = np.mean(l_list[i])
    l_median = l_list[i][round(len(l_list[i]) / 2)]
    print('%s class, l mean: %f  median: %f' % (classes[i], l_mean, l_median))
    z_min = np.min(z_list[i])
    z_max = np.max(z_list[i])
    print('%s class, z_min: %f  z_max: %f' % (classes[i], z_min, z_max))
    if i in [0]:
        # plot histogram
        plt.hist(x=h_list[i], bins=100)
        plt.suptitle('h')
        plt.show()

        plt.hist(x=w_list[i], bins=100)
        plt.suptitle('w')
        plt.show()

        plt.hist(x=l_list[i], bins=100)
        plt.suptitle('l')
        plt.show()
