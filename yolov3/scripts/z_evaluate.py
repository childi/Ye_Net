import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def bbox_iou(boxes1, boxes2):

    boxes1_area = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
    boxes2_area = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    left_up = np.maximum(boxes1[:2], boxes2[:2])
    right_down = np.minimum(boxes1[2:], boxes2[2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[0] * inter_section[1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * inter_area / union_area

    return iou


classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
z_list = [[] for _ in range(len(classes))]
h_list = [[] for _ in range(len(classes))]
z_pred_list = [[] for _ in range(len(classes))]  # z_pred

# use /ground-truth as gt
ground_truth_files_list = glob.glob('/home/zhangy/yolov3/mAP/ground-truth11z/*.txt')  # /media/personal_data/zhangye/outputs/pyramid_cars_with_aug_example
ground_truth_files_list.sort()
for txt_file in ground_truth_files_list:
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    GT_path = os.path.join('/home/zhangy/yolov3/mAP/ground-truth11z/' + file_id + ".txt")  # '000091'
    with open(GT_path, 'r') as gt:
        gt = gt.readlines()
        gt_file = [line.strip() for line in gt]

        PRED_path = os.path.join('/home/zhangy/yolov3/mAP/predicted_wod/' + file_id + ".txt")  # '%06d.txt' % int(file_id)
        with open(PRED_path, 'r') as pred:
            pred = pred.readlines()
            pred_file = [line.strip() for line in pred]
        for box in gt_file:
            box = box.split()
            iou_list = []
            for pred1 in pred_file:
                pred1 = pred1.split()
                if pred1[0].strip() == box[0].strip():
                    iou = bbox_iou(list(map(float, box[1:5])), list(map(int, pred1[2:6])))
                else:
                    iou = 0
                iou_list.append(iou)
            if len(iou_list) > 0 and max(iou_list) > 0.5:
                # id = iou_list.index(max(iou_list))
                index = [i for i, n in enumerate(iou_list) if n == max(iou_list)]
                dz = 90
                for id in index:
                    z_pred = pred_file[id].split()[6]
                    dz = min(abs(float(z_pred) - float(box[5])), dz)
                z_pred_list[int(classes.index(box[0].strip()))].append(float(z_pred))
                if dz > 20:
                    print(file_id, box[0].strip(), dz)
                else:
                    z_list[int(classes.index(box[0].strip()))].append(dz)
                    # if dz > 1:
                    #     print(file_id, z_pred, dz)

for i in range(len(classes) - 1):
    n = len(z_list[i])
    if n > 0 and i in [0, 3, 5]:
        mean = np.mean(z_list[i])
        # z_list[i].sort()
        plt.hist(x=z_list[i], bins=100)
        plt.show()
        plt.hist(x=z_pred_list[i], bins=100)
        plt.show()
        print('%s class, dz mean: %f' % (classes[i], mean))
