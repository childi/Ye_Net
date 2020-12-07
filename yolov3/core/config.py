#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict
import os
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('avod')[0]

__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = os.path.join(ROOT_PATH, "avod/yolov3/data/classes/kitti.names")
__C.YOLO.ANCHORS                = os.path.join(ROOT_PATH, "avod/yolov3/data/anchors/basline_anchors.txt")
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "deconv"  # "resize"
__C.YOLO.ORIGINAL_WEIGHT        = os.path.join(ROOT_PATH, "avod/yolov3/checkpoint/yolov3_coco.ckpt")
__C.YOLO.DEMO_WEIGHT            = os.path.join(ROOT_PATH, "avod/yolov3/checkpoint/yolov3_coco_demo.ckpt")

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = os.path.join(ROOT_PATH, "avod/yolov3/data/dataset/kitti_train.txt")
__C.TRAIN.BATCH_SIZE            = 2  # 6
__C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416]  # , 448, 480, 512, 544, 576, 608
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 100
__C.TRAIN.INITIAL_WEIGHT        = '/media/personal_data/zhangye/yolov3_ckpt/yolov3_test_loss='

# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = os.path.join(ROOT_PATH, "avod/yolov3/data/dataset/kitti_test.txt")
__C.TEST.BATCH_SIZE             = 1  # 2
__C.TEST.INPUT_SIZE             = 416  # 544
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = False
__C.TEST.WRITE_IMAGE_PATH       = os.path.join(ROOT_PATH, "avod/yolov3/data/detection/")
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = ""
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.3
__C.TEST.IOU_THRESHOLD          = 0.45

# "/media/personal_data/zhangye/yolov3_ckpt0827/yolov3_test_loss=13.9093.ckpt-61"
# zz,train1,/media/personal_data/zhangye/yolov3_ckpts/yolov3_test_loss=17.4437.ckpt-76
# 3d_2d,train,/media/personal_data/zhangye/yolov3_ckpt/yolov3_test_loss=16.8527.ckpt-93
