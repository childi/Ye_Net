#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 10:47:03
#   Description :
#
# ================================================================

import numpy as np
import tensorflow as tf
# import torch
import sys
sys.path.append('/home/zhangy/yolov3')
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, trainable, label_sbbox, label_mbbox, label_lbbox, avod=None, pred2d=True):

        self.trainable        = True
        if trainable is False:
            self.trainable        = trainable
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class        = 3  # len(self.classes)
        if avod is not None:
            self.num_class    = 3
        self.strides          = np.array(cfg.YOLO.STRIDES)
        self.anchors          = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD

        # try:
        self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(input_data, avod=avod)
        # except:
        #     raise NotImplementedError("Can not build up yolov3 network!")

        if pred2d is True:
            with tf.variable_scope('pred_sbbox'):
                self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0], label_sbbox)

            with tf.variable_scope('pred_mbbox'):
                self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1], label_mbbox)

            with tf.variable_scope('pred_lbbox'):
                self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2], label_lbbox)

    def __build_nework(self, input_data, avod=None):

        source, source_2, source_4, route_1, route_2, darknet_fe = backbone.darknet53(input_data, self.trainable)

        input_data = common.convolutional(darknet_fe, (1, 1, 1024,  512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 6)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1,  512,  256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512),  self.trainable, name='conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 6)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(self.num_class + 6)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        # upsample*3
        if avod is not None:
            with tf.variable_scope('upsample_layer'):
                # input_data = common.convolutional(darknet_fe, (1, 1, 1024, 256), self.trainable, 'conv69')
                # input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)
                # input_data = tf.concat([input_data, route_2], axis=-1, name='concat0')
                # input_data = common.convolutional(input_data, (1, 1, 768, 128), self.trainable, 'pyramid_fusion0')
                # input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)
                # input_data = tf.concat([input_data, route_1], axis=-1, name='concat1')
                # input_data = common.convolutional(input_data, (1, 1, 384, 64), self.trainable, 'pyramid_fusion1')
                input_data = common.convolutional(input_data, (1, 1, 128, 64), self.trainable, 'conv69')

                input_data = common.upsample(input_data, name='upsample2', method=self.upsample_method)
                input_data = tf.concat([input_data, source_4], axis=-1, name='concat2')
                input_data = common.convolutional(input_data, (1, 1, 192, 32), self.trainable, 'pyramid_fusion2')
                input_data = common.upsample(input_data, name='upsample3', method=self.upsample_method)
                input_data = tf.concat([input_data, source_2], axis=-1, name='concat3')
                input_data = common.convolutional(input_data, (1, 1, 96, 16), self.trainable, 'pyramid_fusion3')
                input_data = common.upsample(input_data, name='upsample4', method=self.upsample_method)
                input_data = tf.concat([input_data, source], axis=-1, name='concat4')
                input_data = common.convolutional(input_data, (1, 1, 48, 32), self.trainable, 'pyramid_fusion4')
                self.feature_maps = input_data

        return conv_lbbox, conv_mbbox, conv_sbbox

    def decode(self, conv_output, anchors, stride, label_bbox):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, 3 * output_size, anchor_per_scale,
                                               6 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        # conv_raw_alpha_cos = conv_output[:, :, :, :, 5:6]
        # conv_raw_alpha_sin = conv_output[:, :, :, :, 6:7]
        conv_raw_z = conv_output[:, :, :, :, 5:6]
        conv_raw_prob = conv_output[:, :, :, :, 6:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, 3 * output_size])
        x = tf.tile(tf.range(3 * output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = label_bbox[:, :, :, :, 2:4]
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        # pred_alpha = tf.atan2(conv_raw_alpha_sin, conv_raw_alpha_cos)
        pred_z = tf.sigmoid(conv_raw_z) * 90 - 2
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_z, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def WeightedSmoothL1Loss(self, target_tensor, prediction_tensor, weight=1):
        """Compute loss function.
        Args:
            prediction_tensor: A float tensor  representing the (encoded) predicted
                locations of objects.
            target_tensor: A float tensor  representing the regression targets
        Returns:
          loss: an anchorwise tensor  representing the value of the loss function
        """
        diff = prediction_tensor - target_tensor
        abs_diff = tf.abs(diff)
        abs_diff_lt_1 = tf.less(abs_diff, 1)

        anchorwise_smooth_l1norm = tf.reduce_sum(
            tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
            axis=-1, keepdims=True) * weight
        return anchorwise_smooth_l1norm

    def tf_orientation_to_angle_vector(self, orientations_tensor):
        x = tf.cos(orientations_tensor)
        y = tf.sin(orientations_tensor)
        return tf.concat([x, y], axis=-1)

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        # union_area = boxes1_area + boxes2_area - inter_area
        union_area = tf.maximum(boxes1_area + boxes2_area - inter_area, 0.0001)
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0001)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        # union_area = boxes1_area + boxes2_area - inter_area
        union_area = tf.maximum(boxes1_area + boxes2_area - inter_area, 0.0001)
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, 3 * output_size,
                                 self.anchor_per_scale, 6 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        # conv_raw_alpha_angle = conv[:, :, :, :, 5:7]
        conv_raw_z = conv[:, :, :, :, 5:6]
        conv_raw_prob = conv[:, :, :, :, 6:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]
        pred_z        = pred[:, :, :, :, 5:6]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        # label_alpha   = label[:, :, :, :, 5:6]
        label_z       = label[:, :, :, :, 5:6]
        label_prob    = label[:, :, :, :, 6:]

        # # 2D xy loss
        pred_xy = pred_xywh[:, :, :, :, 0:2]
        pred_wh = label_xywh[:, :, :, :, 2:4]  # Don't predict wh-->use label wh as predected
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        # label_angle = self.tf_orientation_to_angle_vector(label_alpha)
        # alpha_loss = respond_bbox * (self.WeightedSmoothL1Loss(label_angle, conv_raw_alpha_angle))
        conv_norm_z = tf.sigmoid(conv_raw_z)
        label_norm_z = (label_z + 2) / 90
        z_loss = respond_bbox * (self.WeightedSmoothL1Loss(label_norm_z, conv_norm_z))

        # giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        xy_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
        # alpha_loss = tf.reduce_mean(tf.reduce_sum(alpha_loss, axis=[1,2,3,4]))
        z_loss = tf.reduce_mean(tf.reduce_sum(z_loss, axis=[1,2,3,4]))  # tf.reduce_mean => / batch size

        return xy_loss, conf_loss, prob_loss, z_loss

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors = self.anchors[0], stride = self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors = self.anchors[1], stride = self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.name_scope('xy_loss'):
            xy_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        with tf.name_scope('z_loss'):
            z_loss = loss_sbbox[3] + loss_mbbox[3] + loss_lbbox[3]

        return xy_loss, conf_loss, prob_loss, z_loss


class DARKNET(object):

    def __init__(self, input_data, trainable):

        self.trainable        = trainable
        # self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        # self.num_class        = 3  # len(self.classes)  # for avod, when class=car
        # self.strides          = np.array(cfg.YOLO.STRIDES)
        # self.anchors          = utils.get_anchors(cfg.YOLO.ANCHORS)
        # self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        # self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD

        source, source_2, source_4, route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)

        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        # input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        # input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')

        input_data = common.convolutional(input_data, (1, 1,  512,  256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        # input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        # input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        # input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        # input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        # upsample*3
        input_data = common.convolutional(input_data, (1, 1, 128, 64), self.trainable, 'conv69')
        input_data = common.upsample(input_data, name='upsample2', method=self.upsample_method)
        input_data = tf.concat([input_data, source_4], axis=-1, name='concat2')
        input_data = common.convolutional(input_data, (1, 1, 192, 32), self.trainable, 'pyramid_fusion2')
        input_data = common.upsample(input_data, name='upsample3', method=self.upsample_method)
        input_data = tf.concat([input_data, source_2], axis=-1, name='concat3')
        input_data = common.convolutional(input_data, (1, 1, 96, 16), self.trainable, 'pyramid_fusion3')
        input_data = common.upsample(input_data, name='upsample4', method=self.upsample_method)
        input_data = tf.concat([input_data, source], axis=-1, name='concat4')
        input_data = common.convolutional(input_data, (1, 1, 48, 32), self.trainable, 'pyramid_fusion4')
        self.feature_maps = input_data
