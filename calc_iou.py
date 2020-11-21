from obj_process import ObjectLabel
import cv2
import math
import numpy as np


def cal_iou_overlook(pred_boxes, true_boxes):
    r1 = ((pred_boxes[0], pred_boxes[1]), (pred_boxes[2], pred_boxes[3]), pred_boxes[4]*180.0)
    r2 = ((true_boxes[0], true_boxes[1]), (true_boxes[2], true_boxes[3]), true_boxes[4]*180.0)

    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
    else:
        int_area = 0.0

    int_area = float(int_area)

    return int_area


def calc_iou_3d(obj_gt, obj_pred):

    iou_overlook = cal_iou_overlook([obj_pred.t[0], obj_pred.t[2], obj_pred.l, obj_pred.w, obj_pred.ry/math.pi],
                                    [obj_gt.t[0], obj_gt.t[2], obj_gt.l, obj_gt.w, obj_gt.ry/math.pi])
    pred_h_min = obj_pred.t[1] - obj_pred.h
    pred_h_max = obj_pred.t[1]
    true_h_min = obj_gt.t[1] - obj_gt.h
    true_h_max = obj_gt.t[1]
    h_min_1 = np.maximum(pred_h_min, true_h_min)
    h_max_2 = np.minimum(pred_h_max, true_h_max)
    h_dif = np.maximum(0.0, h_max_2 - h_min_1)
    intersect_area = iou_overlook * h_dif

    pred_box_area = obj_pred.w * obj_pred.l * obj_pred.h
    true_box_area = obj_gt.w * obj_gt.l * obj_gt.h

    iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

    return iou
