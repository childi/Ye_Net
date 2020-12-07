import cv2
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
from collections import Counter

import yolov3.core.utils as utils
from avod.core import anchor_projector
from avod.core import box_list
from avod.core import box_list_ops
from avod.core import box_3d_encoder
from avod.core import box_3d_projector
# from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils

# import evaluate
# base = evaluate.YoloTest()

# plt.switch_backend('agg')

class GS3D(object):
    '''
    yolo-->proposal(prps)
    gs3d-->general(gen)
    avod-->predicted(pred)
    3d projected to img/bev-->2img/2bev;2d in img/bev-->img/bev
    input-->dir;;output-->path
    '''
    def __init__(self):
        self.classes = ['Car', 'Pedestrian', 'Cyclist']
        self.gt2d_dir = "../../yolov3/data/dataset/kitti_test.txt"
        # TODO:When proposal changes from gt to predicted, 2d path=base.write_image_path
        self.write_prps_img_path = '../../yolov3/data/detection2d_img/'
        self.write_genpredgt3d_2bev_path = '../../yolov3/data/detection3d_2bev/'
        self.write_gengt3d_img_path = '../../yolov3/data/detection3d_img/'
        self.write_prps_img_genpredgt3d_2img_path = './yolov3/data/detection_img/'
        self.calib_dir = '/media/dataset/Kitti/object/training/calib/'
        self.label_dir = '/media/dataset/Kitti/object/training/label_2/'
        self.prps_dir = '../../yolov3/mAP/mapp/predicted11z_0.1/'
        # self.prps_dir01 = 'yolov3/mAP/predicted0.2/'
        # self.pred_dir = 'avod/data/outputs/pyramid_cars_with_aug_example/' \
        #                 'predictions/kitti_native_eval/0.1_0830fp/249121/data'
        # self.write_gen_2bev_path = 'yolov3/mAP/predicted_bev/'
        self.write_gen_2native_path = 'avod/data/outputs/pyramid_cars_with_aug_example/predictions/' \
                                      'kitti_native_eval/0.1_gen/32zz_all1/'

    def proposal2bbox(self, box2, classid, stereo_calib, sample_name, score=None, z=None,
                      filter_class=None, alpha=None, aug=[]):
        k = np.matrix(stereo_calib)
        lwh0priori = [[4, 1.7, 1.5],
                      [5.5, 1.9, 2.4],
                      [10.3, 2.7, 3.4],
                      [0.9, 0.8, 1.8],
                      [1, 0.56, 1.3],
                      [1.8, 0.57, 1.8],
                      [18, 2.6, 3.5],
                      [3, 1.3, 1.6]]
        lwh_priori = [[4, 4, 1.5],
                      [0.9, 0.9, 1.8],
                      [1.8, 1.8, 1.8]]

        bbox_list = []
        box_3d_list = []
        box_3d_score_list = []
        n_list = []
        for n, box in enumerate(box2):
            cls = int(classid[n])
            if filter_class is not None:
                if self.classes[cls] not in filter_class:
                    continue
            if alpha is not None:
                lwh = lwh0priori[cls]
                alphan = float(alpha[n])
                # inside = abs(lwh[0] * math.sin(alphan) + lwh[1] * math.cos(alphan)) / 2
            else:
                lwh = lwh_priori[cls]
                # inside = [1.9, 0, 0, 0.45, 0, 0.4, 0, 0][cls]  # change without count

            inside = [1.9, 0, 0, 0.45, 0, 0.4, 0, 0][cls]  # change without count
            # inside = [1.4, 0, 0, 0.02, 0, 0.2, 0, 0][cls]  # change with count
            # # box = [x, y, l, w]
            # center_t_2d = np.matrix([box[0], box[1]-box[3]/2, 1]).reshape([3, 1])
            # center_b_2d = np.matrix([box[0], box[1]+box[3]/2, 1]).reshape([3, 1])
            # # box = [xmin, ymin, xmax, ymax]
            # center_t_2d = np.matrix([(box[0] + box[2]) / 2, box[1], 1]).reshape([3, 1])
            # center_b_2d = np.matrix([(box[0] + box[2]) / 2, box[3], 1]).reshape([3, 1])
            center_2d = np.matrix([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1]).reshape([3, 1])
            # center_t_3d = k.I * center_t_2d
            # center_b_3d = k.I * center_b_2d
            center_3d = k.I * center_2d

            d_list = []
            # 11similar triangles from default height and 2Dbox_height
            # d = lwh[2] / (center_b_3d[1] - center_t_3d[1])
            # d_list.append(d[0, 0])
            # 22from dense depth
            # denth_image = np.load('/media/dataset/Kitti/object/training/denseDepth_jointBF/%06d.npy' % int(sample_name))
            denth_image = np.load('/media/dataset/Kitti/object/training/dense_map_only_lidar/%06d.npy' % int(sample_name))
            # denth_image = np.squeeze(np.load('/media/dataset/Kitti/object/training/l_depth/%06d.npy' % int(sample_name)))
            # denth_image = np.squeeze(np.load('/media/dataset/Kitti/object/training/sl_depth_fullsize/%06d.npy' % int(sample_name)))
            # denth_image = np.squeeze(np.load('/media/dataset/Kitti/object/training/rl_depth/%06d.npy' % int(sample_name)))
            # denth_image = cv2.imread('/media/dataset/Kitti/object/training/FSLD/%06d.png' % int(sample_name), -1) / 255.
            # hd, wd = map(int, denth_image.shape)
            # bgr_image = np.array(cv2.imread('/media/dataset/Kitti/object/training/image_2/%06d.png' % int(sample_name)))
            # h, w, _ = map(int, bgr_image.shape)
            # if wd < w or hd < h:
            #     # print(image_first_name)
            #     tmp = np.zeros((h, w))
            #     tmp[h-hd:, :wd] = denth_image
            #     tmp[:, wd:] = np.tile(tmp[:, wd-1][:, np.newaxis], (w-wd))
            #     denth_image = tmp
            h, w = denth_image.shape
            denth_part = denth_image[max(0, int(box[1])):min(h, int(box[3])), max(0, int(box[0])):min(h, int(box[2]))]
            denth_part_size = denth_part.shape[0]*denth_part.shape[1]
            # # median
            # d_shelter = np.median(denth_part)
            # # mode
            # count = np.bincount([round(i) for i in denth_part.flatten()], minlength=91)  # 0-80 # change with inside
            count = np.bincount(denth_part.flatten().astype(int), minlength=91)  # 0-80
            d_shelter = np.argmax(count)
            d_list.append(d_shelter+inside)
            n_list.append(n)
            count[d_shelter-2:d_shelter+3] = 0
            d_sheltered = np.argmax(count)
            if count[d_sheltered] > int(denth_part_size/20):
                d_list.append(d_sheltered+inside)
                n_list.append(n)
            # combine with pred z
            if z is not None:
                pred_z = float(z[n])
                pred_d = min(pred_z-inside, 90)
                if count[round(pred_d)] > int(denth_part_size/40) or (cls == int(0) or cls == int(5)):
                    d_list.append(pred_z)
                    n_list.append(n)
                elif abs(pred_d - d_sheltered) > 2 and abs(pred_d - d_shelter) > 2:
                    front = max(int(pred_d) - 2, 0)
                    back = min(int(pred_d) + 10, 80)
                    if pred_d > d_sheltered and pred_d > d_shelter:
                        front = max(d_sheltered, d_shelter) + 3
                    elif pred_d < d_sheltered and pred_d < d_shelter:
                        back = min(d_sheltered, d_shelter) - 2
                    else:
                        front = min(d_sheltered, d_shelter) + 3
                        back = max(d_sheltered, d_shelter) - 2
                    if front < back:
                        d_sheltereded = np.argmax(count[front:back]) + front
                        if count[d_sheltereded] > int(denth_part_size/40):
                            d_list.append(d_sheltereded+inside)
                            n_list.append(n)
            # 33from pred together with 2Dbbox
            # if z is not None:
            #     d = float(z[n])
            #     d_list.append(d)  # because label_z=int(z_gt)
            #     n_list.extend([n])

            for d in d_list:
                # center_t = np.multiply(d, center_t_3d)
                # center_b = np.multiply(d, center_b_3d)
                # box_center = (center_t + center_b) / 2  # The center of the whole
                center = np.multiply(d, center_3d)
                box_center = center
                # box3 = np.delete(center_b, 3, 0)  # The center of the upper surface

                bbox = obj_utils.ObjectLabel()
                bbox.h = float(lwh[2])
                bbox.w = float(lwh[1])
                bbox.l = float(lwh[0])
                # bbox.t = (float(box3[0]), float(box3[1]), float(box3[2]))
                bbox.t = (float(box_center[0]), float(box_center[1] + lwh[2]/2), float(box_center[2]))
                bbox.occlusion = int(1)  # occlusion is used specifically to distinguish gen3D from multiple3D

                ry = 0
                if alpha is not None:
                    ry = float(alpha[n]) + math.atan2(box_center[0], box_center[2])
                    bbox.ry = ry
                bbox_list.append(bbox)

                box_3d = [float(box_center[0]), float(box_center[1] + lwh[2]/2), float(box_center[2]), lwh[0], lwh[1], lwh[2], ry]
                box_3d_list.append(box_3d)
                # box_3d_score_list.append(score[n])

                # if score is not None:
                #     with open(self.gen_2bev, 'a') as f:
                #         coor = np.array(box[:4], dtype=np.int32)
                #         class_name = self.classes[cls]
                #         sco = score[n]
                #         xmin, ymin, xmax, ymax = list(map(str, coor))
                #         zz = str(d)
                #         bbox_mess = ' '.join([class_name, sco, xmin, ymin, xmax, ymax, zz]) + '\n'
                #         f.write(bbox_mess)
                #         print('\t' + str(bbox_mess).strip())

                if 'multiple' in aug or (len(box2) == 1 and len(d_list) == 1):
                    box_3d2obj_list = []
                    degree = abs(ry)
                    dz = (lwh[0] / 2) * math.sin(degree) + (lwh[1] / 2) * math.cos(degree)
                    dx = (lwh[0] / 2) * math.cos(degree) + (lwh[1] / 2) * math.sin(degree)
                    # front
                    box_3d_f = [float(box_center[0]), float(box_center[1] + lwh[2]/2), float(box_center[2] - dz),
                                lwh[0], lwh[1], lwh[2], ry]
                    box_3d_list.append(box_3d_f)
                    # behind
                    box_3d_b = [float(box_center[0]), float(box_center[1] + lwh[2]/2), float(box_center[2] + dz),
                                lwh[0], lwh[1], lwh[2], ry]
                    box_3d_list.append(box_3d_b)
                    # left
                    box_3d_l = [float(box_center[0] - dx), float(box_center[1] + lwh[2]/2), float(box_center[2]),
                                lwh[0], lwh[1], lwh[2], ry]
                    box_3d_list.append(box_3d_l)
                    # right
                    box_3d_r = [float(box_center[0] + dx), float(box_center[1] + lwh[2]/2), float(box_center[2]),
                                lwh[0], lwh[1], lwh[2], ry]
                    box_3d_list.append(box_3d_r)
                    # box_3d_score_list.extend([score[n]] * 4)
                    n_list.extend([n, n, n, n])

                    # box_3d2obj_list.append(box_3d_f)
                    # box_3d2obj_list.append(box_3d_b)
                    # box_3d2obj_list.append(box_3d_l)
                    # box_3d2obj_list.append(box_3d_r)
                    #
                    # if len(box_3d2obj_list) != 0:
                    #     for box3d in box_3d2obj_list:
                    #         box2obj = obj_utils.ObjectLabel()
                    #         box2obj.h = float(box3d[5])
                    #         box2obj.w = float(box3d[4])
                    #         box2obj.l = float(box3d[3])
                    #         box2obj.t = (float(box3d[0]), float(box3d[1]), float(box3d[2]))
                    #         box2obj.ry = float(box3d[6])
                    #         box2obj.occlusion = int(2)
                    #         # occlusion is used specifically to distinguish gen3D from multiple3D
                    #         bbox_list.append(box2obj)

                # if alpha is None and 'multiscale' in aug:
                #     lwh0 = lwh0priori[cls]
                #     box_3d1 = [float(box_center[0]), float(box_center[1] + lwh[2]/2), float(box_center[2]),
                #                lwh0[0], lwh0[1], lwh0[2], ry]
                #     box_3d2 = [float(box_center[0]), float(box_center[1] + lwh[2]/2), float(box_center[2]),
                #                lwh0[1], lwh0[0], lwh0[2], ry]
                #     box_3d_list.append(box_3d1)
                #     box_3d_list.append(box_3d2)
                #     if 'multiple' in aug:
                #         # front
                #         box_3d_f1 = [float(box_center[0]), float(box_center[1] + lwh[2]/2), float(box_center[2] - lwh[1] / 2),
                #                      lwh0[0], lwh0[1], lwh0[2], ry]
                #         box_3d_f2 = [float(box_center[0]), float(box_center[1] + lwh[2]/2), float(box_center[2] - lwh[1] / 2),
                #                      lwh0[1], lwh0[0], lwh0[2], ry]
                #         box_3d_list.append(box_3d_f1)
                #         box_3d_list.append(box_3d_f2)
                #         # behind
                #         box_3d_b1 = [float(box_center[0]), float(box_center[1] + lwh[2]/2), float(box_center[2] + lwh[1] / 2),
                #                      lwh0[0], lwh0[1], lwh0[2], ry]
                #         box_3d_b2 = [float(box_center[0]), float(box_center[1] + lwh[2]/2), float(box_center[2] + lwh[1] / 2),
                #                      lwh0[1], lwh0[0], lwh0[2], ry]
                #
                #         box_3d_list.append(box_3d_b1)
                #         box_3d_list.append(box_3d_b2)
                #         # left
                #         box_3d_l1 = [float(box_center[0] - lwh[0] / 2), float(box_center[1] + lwh[2]/2), float(box_center[2]),
                #                      lwh0[0], lwh0[1], lwh0[2], ry]
                #         box_3d_l2 = [float(box_center[0] - lwh[0] / 2), float(box_center[1] + lwh[2]/2), float(box_center[2]),
                #                      lwh0[1], lwh0[0], lwh0[2], ry]
                #         box_3d_list.append(box_3d_l1)
                #         box_3d_list.append(box_3d_l2)
                #         # right
                #         box_3d_r1 = [float(box_center[0] + lwh[0] / 2), float(box_center[1] + lwh[2]/2), float(box_center[2]),
                #                      lwh0[0], lwh0[1], lwh0[2], ry]
                #         box_3d_r2 = [float(box_center[0] + lwh[0] / 2), float(box_center[1] + lwh[2]/2), float(box_center[2]),
                #                      lwh0[1], lwh0[0], lwh0[2], ry]
                #         box_3d_list.append(box_3d_r1)
                #         box_3d_list.append(box_3d_r2)

        # If the prediction category is empty in this sample
        if len(box_3d_list) == 0:
            box_3d_list = np.concatenate((np.random.randint(0, 40, (3, 2)), np.ones((3, 4)), np.zeros((3, 1))), axis=1)
            # box_3d_score_list.extend([0, 0, 0])
            for box3d in box_3d_list:
                box2obj = obj_utils.ObjectLabel()
                box2obj.h = float(box3d[5])
                box2obj.w = float(box3d[4])
                box2obj.l = float(box3d[3])
                box2obj.t = (float(box3d[0]), float(box3d[1]), float(box3d[2]))
                box2obj.ry = float(box3d[6])
                box2obj.occlusion = int(2)  # occlusion is used specifically to distinguish gen3D from multiple3D
                bbox_list.append(box2obj)
            # print(sample_name, '!!!!')
            n_list.extend([-1, -1, -1])

        return bbox_list, np.array(box_3d_list), n_list

    def draw3dbox_xy(self, image, p2, image_dir, img_idx, objects_gen, objects_gt):
        # Run Visualization Function
        def set_plot_limits(axes, image):
            # Set the plot limits to the size of the image, y is inverted
            axes.set_xlim(0, image.shape[1])
            axes.set_ylim(image.shape[0], 0)

        # Create the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9.15), sharex=True)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.0)

        # plot images
        ax1.imshow(image)
        ax2.imshow(image)

        set_plot_limits(ax1, image)
        set_plot_limits(ax2, image)

        # plt.show(block=False)

        # For all annotated objects
        # for obj in objects_gen:
        #     # Draw gt 3D boxes
        #     vis_utils.draw_box_3d(ax1, obj, p2)
        for obj in objects_gt:
            # Draw gen 3D boxes
            vis_utils.draw_box_3d(ax2, obj, p2)


        gen_3d_path = os.path.join(self.write_gengt3d_img_path, str(img_idx) + '.png')
        plt.savefig(gen_3d_path)

        # Render results
        plt.draw()
        plt.show()

    def draw3dbox_xz(self, img_idx, objects_gen, objects_gt, boxes_pred):
        # Run Visualization Function
        def set_plot_limits(axes, shape):
            # Set the plot limits to the size of the image, y is inverted
            axes.set_xlim(- shape[1] / 2, shape[1] / 2)
            axes.set_ylim(0, shape[0])

        # Create the figure
        fig, ax = plt.subplots(1, 1, sharex=True)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.0)

        shape = [70, 80]
        set_plot_limits(ax, shape)

        # plt.show(block=False)

        # For all annotated objects
        for obj in objects_gen:
            # Draw gen 3D boxes
            vis_utils.draw_rot_rec(ax, obj, obj_mode=True, color='k')
        for obj in objects_gt:
            # Draw gt 3D boxes
            vis_utils.draw_rot_rec(ax, obj, obj_mode=True, linewidth=1, color='r')
        for box in boxes_pred:
            # Draw pred 3D boxes
            vis_utils.draw_rot_rec(ax, box, obj_mode=False, linewidth=1, color='g')  #


        gen_bev_path = os.path.join(self.write_genpredgt3d_2bev_path, str(img_idx) + '.png')
        plt.savefig(gen_bev_path)

        # Render results
        plt.draw()
        plt.show()

    def calibration(self, p2, r0_rect, tr_velodyne_to_cam):
        # Pad the r0_rect matrix to a 4x4
        r0_rect_mat = r0_rect
        r0_rect_mat = np.pad(r0_rect_mat, ((0, 1), (0, 1)),
                             'constant', constant_values=0)
        r0_rect_mat[3, 3] = 1

        # Pad the tr_vel_to_cam matrix to a 4x4
        tf_mat = tr_velodyne_to_cam
        tf_mat = np.pad(tf_mat, ((0, 1), (0, 0)),
                        'constant', constant_values=0)
        tf_mat[3, 3] = 1

        # p_cam = P2 * R0_rect * Tr_velo_to_cam * p_velo
        rectified = np.dot(r0_rect_mat, tf_mat)
        rectified = np.dot(p2, rectified)
        return rectified

    def genbev(self, box_gen):
        bev2dbox = obj_utils.ObjectLabel()
        # bev(xz): x->x, y->z
        bev2dbox.x1 = box_gen.t[0] - box_gen.l / 2
        bev2dbox.y1 = box_gen.t[2] - box_gen.w / 2
        bev2dbox.x2 = box_gen.t[0] + box_gen.l / 2
        bev2dbox.y2 = box_gen.t[2] - box_gen.w / 2
        return bev2dbox

    def iou_in_bev(self, box_3d_list, box_gt, n_list, yolo_gt_len):
        bev_proposal_boxes, _ = anchor_projector.project_to_bev(
            box_3d_list[:, 0:6], [[-40, 40], [0, 70]])
        bev_proposal_boxes_tf_order = anchor_projector.reorder_projected_boxes(bev_proposal_boxes)

        label_boxes_3d = np.asarray(
            [box_3d_encoder.object_label_to_box_3d(obj_label) for obj_label in box_gt])
        label_anchors = box_3d_encoder.box_3d_to_anchor(label_boxes_3d, ortho_rotate=True)
        bev_anchor_boxes_gt, _ = anchor_projector.project_to_bev(
            label_anchors, [[-40, 40], [0, 70]])
        bev_anchor_boxes_gt_tf_order = anchor_projector.reorder_projected_boxes(bev_anchor_boxes_gt)

        all_ious = self.iou(bev_anchor_boxes_gt_tf_order, bev_proposal_boxes_tf_order)
        max_ious = np.max(all_ious, axis=0)
        n = 0
        z = 0
        tool = [False for i in max_ious]
        while n < len(max_ious):
            too = [False for i in max_ious]
            for i, x in enumerate(n_list):
                if x == n_list[n]:
                    too[i] = True
                    z = i
            rightiou = np.argmax(max_ious[too]) + n
            n = z + 1
            tool[rightiou] = True
        iou_threshold = heapq.nlargest(yolo_gt_len, max_ious[tool])
        rightn = len(max_ious[tool])
        return iou_threshold, rightn

    def iou(self, boxes1, boxes2):
        y_min1, x_min1, y_max1, x_max1 = np.split(
            boxes1, 4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = np.split(
            boxes2, 4, axis=1)
        all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
        all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
        intersect_heights = np.maximum(
            0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
        all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
        intersect_widths = np.maximum(
            0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        intersections = intersect_heights * intersect_widths

        areas1 = np.squeeze((y_max1 - y_min1) * (x_max1 - x_min1), 1)
        areas2 = np.squeeze((y_max2 - y_min2) * (x_max2 - x_min2), 1)
        unions = (np.expand_dims(areas1, 1) + np.expand_dims(areas2, 0) - intersections)
        return np.where(np.equal(intersections, 0.0),
                        np.zeros_like(intersections), np.true_divide(intersections, unions))

    def build(self):
        if os.path.exists(self.write_prps_img_path): shutil.rmtree(self.write_prps_img_path)
        os.mkdir(self.write_prps_img_path)
        if os.path.exists(self.write_genpredgt3d_2bev_path): shutil.rmtree(self.write_genpredgt3d_2bev_path)
        os.mkdir(self.write_genpredgt3d_2bev_path)
        if os.path.exists(self.write_gengt3d_img_path): shutil.rmtree(self.write_gengt3d_img_path)
        os.mkdir(self.write_gengt3d_img_path)
        # if os.path.exists(self.write_gen_2bev_path): shutil.rmtree(self.write_gen_2bev_path)
        # os.mkdir(self.write_gen_2bev_path)
        # if os.path.exists(self.write_gen_2native_path): shutil.rmtree(self.write_gen_2native_path)
        # os.mkdir(self.write_gen_2native_path)
        # os.mkdir(self.write_gen_2native_path + 'data/')
        iou_threshold_list = []
        # f=open("yolov3/data/dataset/kitti_train3d_2d.txt", 'a')
        with open(self.gt2d_dir, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image_first_name = image_name.split('.')[0]
                # image_first_name = ['000000', '000167', '000265', '000287', '000308', '000782', '000943'][num]  #
                # image_name = image_first_name + '.png'  #
                # image_path = '/media/dataset/Kitti/object/training/image_2/' + image_name  #
                image = cv2.imread(image_path)

                # TODO:proposal
                # 11 kitti gt 2d box(compare with gt 3d)
                bbox_data_gt = np.array([list(map(float, box.split(','))) for box in annotation[1:]])
                if len(bbox_data_gt) == 0:
                    bboxes_gt, classes_gt, alpha_gt= [], [], []
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]  # , alpha_gt, bbox_data_gt[:, 5]
                # proposal = bboxes_gt
                # classes = classes_gt
                # alpha = alpha_gt
                # 22 predicted 2d proposal
                # bboxes_pr = base.predict(image)
                # if len(bboxes_pr) == 0:
                #     break
                # else:
                #     proposal, classes = bboxes_pr[:, :4], bboxes_pr[:, 4]
                # 33 reading predicted boxes-->prps
                box_prps_path = os.path.join(self.prps_dir, str(int(image_first_name)) + '.txt')  # '%06d.txt' % num
                with open(box_prps_path, 'r') as predicted_file:
                    predicted_box = predicted_file.readlines()
                    proposal = []
                    classes = []
                    score = []
                    alpha = []
                    z = []
                    if len(predicted_box) != 0:
                        for box in predicted_box:
                            box2 = list(map(int, box.split()[2:6]))
                            proposal.append(box2)
                            classid = self.classes.index(box.split()[0])
                            classes.append(str(classid))
                            score.append(box.split()[1])
                            # alpha.append(box.split()[7])
                            z.append(box.split()[6])
                # box_prps_path01 = os.path.join(self.prps_dir01, str(int(image_first_name)) + '.txt')  # '%06d.txt' % num
                # with open(box_prps_path01, 'r') as predicted_file:
                #     predicted_box = predicted_file.readlines()
                #     proposal01 = []
                #     classes01 = []
                #     score01 = []
                #     # alpha = []
                #     z01 = []
                #     if len(predicted_box) != 0:
                #         for box in predicted_box:
                #             box2 = list(map(int, box.split()[2:6]))
                #             proposal01.append(box2)
                #             classid = self.classes.index(box.split()[0])
                #             classes01.append(str(classid))
                #             score01.append(box.split()[1])
                #             # alpha.append(box.split()[6])
                #             z01.append(box.split()[6])

                # reading calibration file from kitti dataset
                calib_path = os.path.join(self.calib_dir, image_first_name + '.txt')
                calib_data = []
                with open(calib_path, 'r') as calib_file:
                    for n, line in enumerate(calib_file):
                        calib_data.append(line.strip().split())
                    p2 = np.reshape(list(map(float, calib_data[2][1:])), (3, 4))
                    R0 = np.reshape(list(map(float, calib_data[4][1:])), (3, 3))
                    velo_to_cam = np.reshape(list(map(float, calib_data[5][1:])), (3, 4))
                    calib_file.close()
                # velo2pixel = self.calibration(p2, R0, velo_to_cam)

                # 2d~~~~>3d
                # if score is not None:
                #     self.gen_2bev = os.path.join(self.write_gen_2bev_path, str(image_first_name) + '.txt')
                #     if os.path.exists(self.gen_2bev): shutil.rmtree(self.gen_2bev)
                #     os.mknod(self.gen_2bev)
                # gen
                # box_gen : Center of upper surface
                # box_3d_list: The center of the whole object
                box_gen, box_3d_list, box_gen_score = self.proposal2bbox(
                    proposal, classes, p2, image_first_name, score=score, z=z,
                    filter_class=['Car', 'Pedestrian', 'Cyclist'])  # , alpha=alpha, aug=['multiple']
                # box_gen01, box_3d_list01, cls_gen01 = self.proposal2bbox(
                #     proposal01, classes01, p2, image_first_name, z=z01, filter_class=['Car', 'Pedestrian', 'Cyclist'])

                # gen --> 2img
                # ^target rotation is considered(but the box generated by gs3d has no Angle)
                # h, w, _ = image.shape
                # box_3d_2anchor = box_3d_encoder.box_3d_to_anchor(box_3d_list)
                # gen_2img, _ = anchor_projector.project_to_image_space(box_3d_2anchor, p2, [h, w])

                ######
                # gen --> 2kitti_native
                ######
                # (N, 16) is allocated but only values [4:16] are used
                # kitti_predictions = np.zeros([len(box_3d_2anchor), 16])
                # # Get object types
                # obj_types = ['Car'] * len(box_3d_2anchor)
                # # Alpha (Not computed)
                # kitti_predictions[:, 3] = -10 * np.ones((len(box_3d_2anchor)), dtype=np.int32)
                # # 2D predictions
                # kitti_predictions[:, 4:8] = gen_2img[:, 0:4]
                # # 3D predictions
                # # (l, w, h)
                # kitti_predictions[:, 8] = box_3d_2anchor[:, 4]
                # kitti_predictions[:, 9] = box_3d_2anchor[:, 5]
                # kitti_predictions[:, 10] = box_3d_2anchor[:, 3]
                # # (x, y, z)
                # kitti_predictions[:, 11:14] = box_3d_2anchor[:, 0:3]
                # # (ry, score)
                # # kitti_predictions[:, 14] = box_3d_list[:, 6]
                # kitti_predictions[:, 15] = box_gen_score
                # # Round detections to 3 decimal places
                # kitti_predictions = np.round(kitti_predictions, 3)
                # # Empty Truncation, Occlusion
                # kitti_empty_1 = -1 * np.ones((len(box_3d_list), 2), dtype=np.int32)
                # # Stack 3D predictions text
                # kitti_text_3d = np.column_stack([obj_types, kitti_empty_1, kitti_predictions[:, 3:16]])
                # # Save to text files
                # np.savetxt(self.write_gen_2native_path + 'data/' + image_first_name + '.txt', kitti_text_3d,
                #            newline='\r\n', fmt='%s')

                # 3d
                box_gt = obj_utils.read_labels(self.label_dir, int(image_first_name))
                # gt_3d_list = []
                # for obj in box_gt:
                #     if obj.type in ['Car']:   # , 'Pedestrian', 'Cyclist'
                #         gt_3d = [obj.t[0], obj.t[1], obj.t[2], obj.l, obj.w, obj.h, obj.ry]
                #         gt_3d_list.append(gt_3d)
                # box_3d_gt_2anchor = box_3d_encoder.box_3d_to_anchor(gt_3d_list)
                # gt_2img, _ = anchor_projector.project_to_image_space(box_3d_gt_2anchor, p2, [h, w])
                # gt2d project to 2d-->uesed for gt 2d
                # annotation = '/media/dataset/Kitti/object/training/image_2/%06d.png' % int(image_first_name)
                # for i, gt2img in enumerate(gt_2img):
                #     xmin = str(round(float(gt2img[0])))
                #     xmax = str(round(float(gt2img[2])))
                #     ymin = str(round(float(gt2img[1])))
                #     ymax = str(round(float(gt2img[3])))
                #     z = str(gt_3d_list[i][2])
                #     annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(0), z])  # , alpha
                # print(annotation)
                # f.write(annotation + "\n")

                # avod pred
                box3_avod = []
                # box3d_predicted_path = os.path.join(self.pred_dir, str(image_first_name) + '.txt')  # '%06d.txt' % num
                # with open(box3d_predicted_path, 'r') as predicted_file:
                #     predicted3d_box = predicted_file.readlines()
                #     # --> box2d&box3d
                #     classes_avod = []
                #     box2_avod = []
                #     box3_avod = []
                #     # box3d --> 2img
                #     pred_2img = []
                #     classes_avod_2img = []
                #     if len(predicted3d_box) != 0:
                #         for box3d in predicted3d_box:
                #             classid = self.classes.index(box3d.split()[0])
                #             classes_avod.append(str(classid))
                #             box2 = list(map(float, box3d.split()[4:8]))
                #             box3 = list(map(float, box3d.split()[8:15]))
                #             box2_avod.append(box2)
                #             box3_avod.append(box3)
                #             # ^Consider target rotation
                #             final_pred = list(map(float, box3d.split()[11:14]))
                #             final_pred.extend([float(box3d.split()[10])])
                #             final_pred.extend([float(box3d.split()[9])])
                #             final_pred.extend([float(box3d.split()[8])])
                #             final_pred.extend([float(box3d.split()[14])])
                #             pred_img = box_3d_projector.project_to_image_space(np.array(final_pred), p2, truncate=True,
                #                                                                image_size=[w, h])
                #             if pred_img is not None:
                #                 pred_2img.append(pred_img)
                #                 classes_avod_2img.append(str(classid))

                # validation
                # # 2d
                # image1 = utils.draw2dbox(image, bboxes_gt, classes_gt, colors=[(0, 255, 0)], show_label=None)  # 'gt'
                # image = utils.draw2dbox(image, proposal, classes, colors=[(255, 0, 255)], show_label='prps')
                # image = utils.draw2dbox(image, proposal01, classes01, colors=[(0, 0, 255)], show_label='prps0.3')
                # image = utils.draw2dbox(image, gen_2img, cls_gen, colors=[(0, 0, 255)], show_label='gen2')
                # image = utils.draw2dbox(image, gt_2img, classes_gt, colors=[(0, 255, 255)], show_label='gt2')
                # image = utils.draw2dbox(image, box2_avod, classes_avod, colors=[(255, 0, 0)], show_label='pred')
                # image = utils.draw2dbox(image, pred_2img, classes_avod_2img, colors=[(255, 255, 0)], show_label='pred2')
                # cv2.imwrite(self.write_prps_img_path + image_name, image)

                # iou in bev
                iou_threshold, rightn = self.iou_in_bev(box_3d_list, box_gt, box_gen_score, len(bbox_data_gt))
                print(image_first_name, 'gt_len', iou_threshold, 'prps_len', rightn)
                iou_threshold_list.extend(iou_threshold)

                # self.draw3dbox_xy(image, p2, self.write_prps_img_path, image_first_name, box_gen, box_gt)
                # projected to bev
                # self.draw3dbox_xz(image_first_name, box_gen, box_gt, box3_avod)
                # self.draw3dbox_xz(image_first_name, box_gen, box_gt, box_gen01)

            print(np.mean(iou_threshold_list))
            plt.hist(x=iou_threshold_list, bins=100)
            plt.show()


if __name__ == "__main__":
    GS3D().build()
    # box_path = os.path.join('/media/dataset/Kitti/object', 'train' + 'ing',
    #                         'label_2', '000000' + '.txt')
    # stereo_calib_p2 = np.array([[721.5377, 0.000000, 609.5593, 44.85728],
    #                             [0.000000, 721.5377, 172.8540, 0.2163791],
    #                             [0.000000, 0.000000, 1.000000, 0.002745884]])
    # with open(box_path, 'r') as file:
    #     cla = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
    #     boxes = file.readlines()
    #     if len(boxes) != 0:
    #         proposal = []
    #         classes = []
    #         for box in boxes:
    #             classid = cla.index(box.split()[0])
    #             if classid != 8:
    #                 box2 = list(map(float, box.split()[4:8]))
    #                 proposal.append(box2)
    #                 classes.append(str(classid))
    # box, box3 = GS3D().proposal2bbox(proposal, classes, stereo_calib_p2)
    # print(box3)
