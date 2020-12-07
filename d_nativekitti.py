import os
from obj_process import process_gt, process_pred, ObjectLabel, get_p, map_process, difficulty_process
from calc_iou import calc_iou_3d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from avod.core import proposal2bbox
sys.path.append('./avod/data/outputs/pyramid_cars_with_aug_example/predictions/kitti_native_eval')
from wujx import Calibration
from xiezz import box_from_gt_label, box_from_pred_label, vis_labels

# dir_path = './avod/data/outputs/pyramid_cars_with_aug_example/predictions/kitti_native_eval/0.1_gen/01/data'
dir_path = './avod/data/outputs/pyramid_cars_with_aug_example/predictions/kitti_native_eval/0.1_11z0905/148126/data'
gt_dir_path = '/media/dataset/Kitti/object/training/label_2'
calib_dir_path = '/media/dataset/Kitti/object/training/calib'
files = os.listdir(path=dir_path)
files.sort(key= lambda x:int(x[:-4]))
file_pred_list = []
file_gt_list = []
for file in files:
    obj_pred_list = []
    obj_gt_list = []
    if not os.path.isdir(file):
        f_pred = open(dir_path + '/' + file)
        f_gt = open(gt_dir_path + '/' + file)
        # print(f)
        iter_f = iter(f_pred)
        iter_f_gt = iter(f_gt)
        cam_matrix = get_p(calib_dir_path + '/' + file)
        for line in iter_f:
            # print(line)
            obj = process_pred(line, file)
            # obj = map_process(obj, cam_matrix)
            # if obj is None:
            #     continue
            obj_pred_list.append(obj)
        for line in iter_f_gt:
            obj = process_gt(line, file)
            if obj.type == 'Car' or obj.type == 'Van':
                # obj = map_process(obj, cam_matrix)
                obj = difficulty_process(obj)
                # if obj is None:
                #     continue
                obj_gt_list.append(obj)
    if obj_pred_list != [] and obj_gt_list != []:
        for obj_pred in obj_pred_list:
            iou_list = []
            for obj_gt in obj_gt_list:
                iou = calc_iou_3d(obj_gt, obj_pred)
                if obj_gt.match == 1.0:
                    iou_list.append(0.0)
                else:
                    iou_list.append(iou)
            ovmax = np.max(np.array(iou_list))
            jmax = np.argmax(np.array(iou_list))
            if ovmax != 0.0:
                obj_pred.iou = ovmax
                if obj_gt_list[jmax].iou < ovmax:
                    obj_gt_list[jmax].iou = ovmax
                    obj_gt_list[jmax].match_obj = obj_pred
                obj_pred.difficulty = obj_gt_list[jmax].difficulty
                obj_pred.match_obj = obj_gt_list[jmax]
                # obj_gt_list[jmax].match = 1.0
        file_pred_list.append(obj_pred_list)
        file_gt_list.append(obj_gt_list)
a = 0
obj_num = 0
obj_right_num = 0
# obj_tmp_list = []
for obj_gt_list in file_pred_list:
    hi_scores = []
    filename = files[a].split('.')[0]
    gt_label_path = '/media/dataset/Kitti/object/training/label_2/' + '%s.txt' % filename
    calib_path = gt_label_path.replace('label_2', 'calib')
    calibs = Calibration(calib_path)
    gt_boxes = box_from_gt_label(gt_label_path, calibs)
    pred_label_path = './avod/data/outputs/pyramid_cars_with_aug_example/predictions/' \
                      'kitti_native_eval/0.1_11z0905/148126/data/' + '%s.txt' % filename
    pred_boxes = box_from_pred_label(pred_label_path, calibs)
    for obj_gt in obj_gt_list:
        if obj_gt.difficulty >= 3.0:
            continue
        if obj_gt.type == 'Van':
            continue
        obj_num = obj_num + 1
        if obj_gt.iou >= 0.2 and obj_gt.score >= 0.01:
            obj_right_num = obj_right_num + 1
            # obj_tmp_list.append(obj_gt)
            hi_scores.append(obj_gt.score)
    # proposal2bbox.GS3D().draw3dbox_xz(a, file_gt_list[a], obj_tmp_list, [])
    # img = Image.open('./yolov3/data/detection2d_img/%s.png' % files[a].split('.')[0])
    # plt.figure("%s" % files[a])
    # plt.imshow(img)
    # plt.show()
    if len(hi_scores) != 0:
        hi_boxes = {}
        for i in pred_boxes:
            if pred_boxes[i]['score'] in hi_scores:
                hi_boxes[i] = pred_boxes[i]
        vis_labels(gt_boxes, hi_boxes) # hi_boxes
    a = a+1
print(obj_right_num/obj_num)
# x_d_list = []
# y_d_list = []
# z_d_list = []
# w_d_list = []
# l_d_list = []
# h_d_list = []
# z_list = []
# r_list = []
# r_d_list = []
# alpha_d_list = []
#
# for obj_gt in obj_tmp_list:
#     if obj_gt.match_obj !=  None:
#         x_d = obj_gt.t[0] - obj_gt.match_obj.t[0]
#         y_d = obj_gt.t[1] - obj_gt.match_obj.t[1]
#         z_d = obj_gt.t[2] - obj_gt.match_obj.t[2]
#         r_d = (obj_gt.t[0]**2 + obj_gt.t[2]**2)**(1/2) - (obj_gt.match_obj.t[0]**2 + obj_gt.match_obj.t[2]**2)**(1/2)
#         w_d = obj_gt.w - obj_gt.match_obj.w
#         l_d = obj_gt.l - obj_gt.match_obj.l
#         h_d = obj_gt.h - obj_gt.match_obj.h
#         alpha_d = obj_gt.alpha - obj_gt.match_obj.alpha
#
#         x_d_list.append(abs(x_d))
#         y_d_list.append(abs(y_d))
#         z_d_list.append(abs(z_d))
#         z_list.append(obj_gt.t[2])
#         r_list.append((obj_gt.t[0]**2 + obj_gt.t[2]**2)**(1/2))
#         r_d_list.append(abs(r_d))
#         w_d_list.append(w_d)
#         l_d_list.append(l_d)
#         h_d_list.append(h_d)
#         alpha_d_list.append(alpha_d)
#
# path = './test3/'
#
# plt.hist(x_d_list, bins=500, histtype='bar', rwidth=0.8)
# plt.legend()
# plt.suptitle('xd')
# plt.savefig(path + 'xd.jpg')
# plt.close()
#
# plt.hist(y_d_list, bins=500, histtype='bar', rwidth=0.8)
# plt.legend()
# plt.suptitle('yd')
# plt.savefig(path + 'yd.jpg')
# plt.close()
#
# plt.hist(z_d_list, bins=500, histtype='bar', rwidth=0.8)
# plt.legend()
# plt.suptitle('zd')
# plt.savefig(path + 'zd.jpg')
# plt.close()
#
# plt.hist(r_d_list, bins=500, histtype='bar', rwidth=0.8)
# plt.legend()
# plt.suptitle('rd')
# plt.savefig(path + 'rd.jpg')
# plt.close()
#
# plt.hist(w_d_list, bins=500, histtype='bar', rwidth=0.8)
# plt.legend()
# plt.suptitle('wd')
# plt.savefig(path + 'wd.jpg')
# plt.close()
#
# plt.hist(l_d_list, bins=500, histtype='bar', rwidth=0.8)
# plt.legend()
# plt.suptitle('ld')
# plt.savefig(path + 'ld.jpg')
# plt.close()
#
# plt.hist(h_d_list, bins=500, histtype='bar', rwidth=0.8)
# plt.legend()
# plt.suptitle('hd')
# plt.savefig(path + 'hd.jpg')
# plt.close()
#
# plt.hist(alpha_d_list, bins=500, histtype='bar', rwidth=0.8)
# plt.legend()
# plt.suptitle('alpha_d')
# plt.savefig(path + 'alpha_d.jpg')
# plt.close()
#
# plt.scatter(x_d_list, r_d_list, s=1, c='r', alpha=0.4, label='xd_rd')
# plt.scatter(y_d_list, r_d_list, s=1, c='g', alpha=0.4, label='yd_rd')
# plt.legend()
# plt.savefig(path + 'xd_rd.jpg', dpi=300)
# plt.close()
#
# plt.scatter(z_d_list, r_list, s=1, c='r', alpha=0.4, label='zd_r')
# plt.legend()
# plt.savefig(path + 'zd_r.jpg', dpi=300)
# plt.close()
