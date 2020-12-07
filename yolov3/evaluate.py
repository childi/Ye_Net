#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import yolov3.core.utils as utils
from yolov3.core.config import cfg
from yolov3.core.yolov3_z import YOLOV3
from yolov3.core.dataset import Dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 4], name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')
            self.p2_I = tf.placeholder(dtype=tf.float32, shape=[None, 4, 3], name='input_data')

        model = YOLOV3(self.input_data, self.trainable, self.p2_I)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        # var = tf.trainable_variables()
        # conv_layer = ['conv%d' % (52 + i) for i in range(17)]  # 'conv52'到'conv68'
        # all_name = ['conv_sobj_branch', 'conv_mobj_branch', 'conv_lobj_branch', 'darknet',
        #             'conv_lbbox', 'conv_mbbox', 'conv_sbbox']
        # all_name.extend(conv_layer)
        # var_to_restore = [val for val in var if str(val.name).split('/')[0] in all_name]
        # print('var_to_restore:', len(var_to_restore))
        # saver_in = tf.train.Saver(var_to_restore)
        # saver_in.restore(self.sess, '/media/personal_data/zhangye/previous_results/yolov3_img&depth_z0604/yolov3_test_loss=8.9378.ckpt-1')
        # # qqq = self.sess.run(tf.report_uninitialized_variables())
        # # print('utill not init:', len(qqq))
        # var_to_init = [val for val in var if str(val.name).split('/')[0] not in all_name]
        # print('var_to_init:', len(var_to_init))
        # init_new_vars_op = tf.initialize_variables(var_to_init)
        # self.sess.run(init_new_vars_op)
        # print('yolov3 checkpoints(image)')
        self.saver.restore(self.sess, self.weight_file)
        # uninit_vars = []
        # for var in tf.global_variables():
        #     try:
        #         self.sess.run(var)
        #     except tf.errors.FailedPreconditionError:
        #         uninit_vars.append(var)
        # print(uninit_vars)

    def predict(self, image, p2_I):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, 3 * self.input_size])
        image_data = image_data[np.newaxis, ...]
        p2_I = p2_I[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False,
                self.p2_I: p2_I
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 6 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 6 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 6 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes_inavod(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def evaluate(self):
        predicted_dir_path = './mAP/predicted_norz'
        # ground_truth_dir_path = './mAP/ground-truth_train3d_2d'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        # if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        # os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image_path = os.path.join('/media/dataset/Kitti/object/training/image_2', image_name)
                # image = cv2.imread(image_path)
                image_first_name = image_name.split('.')[0]
                bgr_image = np.array(cv2.imread(image_path))
                rgb_image = bgr_image[..., :: -1]
                # dense_image = np.load('/media/dataset/Kitti/object/training/denseDepth_jointBF/' + image_first_name + '.npy')
                dense_image = np.load('/media/dataset/Kitti/object/training/dense_map_only_lidar/' + image_first_name + '.npy')
                # dense_image = np.squeeze(np.load('/media/dataset/Kitti/object/training/l_depth/' + image_first_name + '.npy'))
                # dense_image = np.squeeze(np.load('/media/dataset/Kitti/object/training/sl_depth_fullsize/' + image_first_name + '.npy'))
                # dense_image = np.squeeze(np.load('/media/dataset/Kitti/object/training/rl_depth/' + image_first_name + '.npy'))
                # dense_image = cv2.imread('/media/dataset/Kitti/object/training/FSLD/' + image_first_name + '.png', -1) / 255.
                # hd, wd = map(int, dense_image.shape)
                # h, w, _ = map(int, rgb_image.shape)
                # if wd < w or hd < h:
                #     # print(image_first_name)
                #     tmp = np.zeros((h, w))
                #     tmp[h-hd:, w-wd:] = dense_image
                #     tmp[:, :w-wd] = np.tile(tmp[:, w-wd][:, np.newaxis], (w-wd))
                #     dense_image = tmp
                image = np.concatenate((rgb_image, dense_image[:, :, np.newaxis]), axis=-1)
                # image = rgb_image
                # bbox_data_gt = np.array([list(map(float, box.split(','))) for box in annotation[1:]])
                # print(image_first_name)

                calib_path = os.path.join('/media/dataset/Kitti/object/training/calib/', image_first_name + '.txt')
                calib_data = []
                with open(calib_path, 'r') as calib_file:
                    for n, line in enumerate(calib_file):
                        calib_data.append(line.strip().split())
                    p2 = np.reshape(list(map(float, calib_data[2][1:])), (3, 4))
                p2_I = np.linalg.pinv(p2)

                # if len(bbox_data_gt) == 0:
                #     bboxes_gt = []
                #     classes_gt = []
                #     z_gt = []
                #     alpha_gt = []
                # else:
                #     bboxes_gt, classes_gt, z_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4], bbox_data_gt[:, 5]
                #     # , alpha_gt  # , bbox_data_gt[:, 6]
                # ground_truth_path = os.path.join(ground_truth_dir_path, str(int(image_first_name)) + '.txt')
                #
                # print('=> ground truth of %s:' % image_name)
                # num_bbox_gt = len(bboxes_gt)
                # with open(ground_truth_path, 'w') as f:
                #     for i in range(num_bbox_gt):
                #         class_name = self.classes[classes_gt[i]]
                #         xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                #         z = str(z_gt[i])
                #         # alpha = str(alpha_gt[i])
                #         bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax, z]) + '\n'  # , alpha
                #         f.write(bbox_mess)
                #         print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path, str(int(image_first_name)) + '.txt')
                bboxes_pr = self.predict(image, p2_I)  # xi, yi, xa, ya, score, cla_id->xi, yi, xa, ya, score, cla_id, alpha

                if self.write_image:
                    image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path+image_name, image)

                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        z = str(bbox[6])
                        # alpha = str(bbox[7])
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax, z]) + '\n'  # , alpha
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())

    def voc_2012_test(self, voc2012_test_path):

        img_inds_file = os.path.join(voc2012_test_path, 'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]

        results_path = 'results/VOC2012/Main'
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path)

        for image_ind in image_inds:
            image_path = os.path.join(voc2012_test_path, 'JPEGImages', image_ind + '.jpg')
            image = cv2.imread(image_path)

            print('predict result of %s:' % image_ind)
            bboxes_pr = self.predict(image)
            for bbox in bboxes_pr:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                with open(os.path.join(results_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())


if __name__ == '__main__': YoloTest().evaluate()



