import numpy as np
import matplotlib.pyplot as plt

n = 0
m = 0
l = 0
alpha_list = []
with open('/home/zhangy/yolov3/data/dataset/kitti_test11.txt', 'r') as f:
    txt = f.readlines()
    for line in txt:
        ann = line.split()[1:]
        for box in ann:
            l = l + 1
            clais = box.split(',')[4]
            if int(clais) == 0:
                n = n + 1
                alpha = box.split(',')[-1]
                alpha_list.append(float(alpha))
                if float(alpha) < -1.57 or float(alpha) > 1.57:
                    m = m + 1
    print('_cla num', n)
    print('alpha min', np.min(alpha_list))
    print('alpha max', np.max(alpha_list))
    print('out alpha', m)
    print('box num', l)
    print('_cla num / sample', n/6733)
    plt.hist(x=alpha_list, bins=100)
    plt.show()
