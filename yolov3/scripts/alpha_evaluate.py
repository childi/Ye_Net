import numpy as np
import os
import math
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
alpha_list = [[] for _ in range(len(classes))]  # dz
alpha_in_list = [[] for _ in range(len(classes))]  # z_gt
alpha_out_list = [[] for _ in range(len(classes))]  # z_gt
alpha_pred_list = [[] for _ in range(len(classes))]  # z_pred
h_list = [[] for _ in range(len(classes))]
# use kitti_test.txt as gt
# with open('../data/dataset/kitti_test.txt', 'r') as annotation_file:
#     for num, line in enumerate(annotation_file):
#         annotation = line.strip().split()
#         bbox_data_gt = np.array([list(map(float, box.split(','))) for box in annotation[1:]])
#
#         PRED_path = os.path.join('../mAP/predicted/', str(num) + '.txt')
#         with open(PRED_path, 'r') as pred:
#             pred = pred.readlines()
#             pred_file = [line.strip() for line in pred]
#         for pred1 in pred_file:
#             pred1 = pred1.split()
#             class_ind = classes.index(pred1[0].strip())
#             iou_list = []
#             for box in bbox_data_gt:
#                 if class_ind == box[4]:
#                     iou = bbox_iou(box[0:4], list(map(int, pred1[2:6])))
#                 else:
#                     continue
#                 iou_list.append(iou)
#             if len(iou_list) > 0:
#                 id = iou_list.index(max(iou_list))
#                 alpha_gt = bbox_data_gt[id-1][5]
#                 alpha = abs(alpha_gt - float(pred1[6]))
#                 alpha_list[int(bbox_data_gt[id-1][4])].append(alpha)

# use /ground-truth as gt
ground_truth_files_list = glob.glob('../mAP/ground-truth11/*.txt')  # /media/personal_data/zhangye/outputs/pyramid_cars_with_aug_example
ground_truth_files_list.sort()
for txt_file in ground_truth_files_list:
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    GT_path = os.path.join('../mAP/ground-truth11/' + file_id + ".txt")
    with open(GT_path, 'r') as gt:
        gt = gt.readlines()
        gt_file = [line.strip() for line in gt]

        PRED_path = os.path.join('../mAP/predicted11/', str(file_id) + '.txt')
        with open(PRED_path, 'r') as pred:
            pred = pred.readlines()
            pred_file = [line.strip() for line in pred]
        # for pred1 in pred_file:
        #     pred1 = pred1.split()
        #     # class_ind = classes.index(pred1[0].strip())
        #     iou_list = []
        #     for box in gt_file:
        #         box = box.split()
        #         if pred1[0].strip() == box[0].strip():
        #             iou = bbox_iou(list(map(float, box[1:5])), list(map(int, pred1[2:6])))
        #         else:
        #             iou = 0
        #         iou_list.append(iou)
        #     if len(iou_list) > 0 and max(iou_list) > 0.5:
        #         id = iou_list.index(max(iou_list))
        #         alpha_gt = gt_file[id].split()[-1]
        #         h_gt = abs(float(gt_file[id].split()[4]) - float(gt_file[id].split()[2]))
        #         h = abs(float(pred1[5]) - float(pred1[3]))
        #         dh = abs(h - h_gt)
        #         dalpha = abs(float(alpha_gt) - float(pred1[7]))
        #         alpha_list[int(classes.index(gt_file[id].split()[0].strip()))].append(dalpha)
        #         h_list[int(classes.index(gt_file[id].split()[0].strip()))].append(dh)
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
                index = [i for i, n in enumerate(iou_list) if n == max(iou_list)]
                dalpha = np.pi
                for id in index:
                    z_pred = pred_file[id].split()[7]
                    dalpha = min(abs(float(z_pred) - float(box[6])), dalpha)
                alpha_list[int(classes.index(box[0].strip()))].append(dalpha)
                alpha_pred_list[int(classes.index(box[0].strip()))].append(float(z_pred))
                if dalpha > np.pi/4:
                    print(file_id, box[6], z_pred)
                    alpha_out_list[int(classes.index(box[0].strip()))].append(float(box[6]))
                else:
                    alpha_in_list[int(classes.index(box[0].strip()))].append(float(box[6]))

for i in range(len(classes) - 1):
    mean = np.mean(alpha_list[i])
    meanin = np.mean(alpha_in_list[i])
    # h_mean = np.mean(h_list[i])
    n = len(alpha_list[i])
    alpha_list[i].sort()
    n4 = 0
    n6 = 0
    n12 = 0
    if n > 0 and i in [0, 3, 5]:
        for a in alpha_list[i]:
            if a < (math.pi / 4):
                n4 = n4 + 1
                if a < (math.pi / 6):
                    n6 = n6 + 1
                    if a < (math.pi / 12):
                        n12 = n12 + 1
        print('%s class, alpha mean: %f, alpha in mean: %f, alpha < π/4: %f, alpha < π/6: %f, alpha < π/12: %f'
              % (classes[i], mean, meanin, n4 / n, n6 / n, n12 / n))
        plt.hist(x=alpha_list[i], bins=100)
        plt.show()
        plt.hist(x=alpha_in_list[i], bins=100)
        plt.show()
        plt.hist(x=alpha_out_list[i], bins=100)
        plt.show()
        plt.hist(x=alpha_pred_list[i], bins=100)
        plt.show()
        # print(h_mean)
