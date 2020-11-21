import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import os
import random
import cv2
import shutil
from PIL import Image
import matplotlib.pyplot as plt

from avod.builders import feature_extractor_builder
from avod.core import anchor_encoder
from avod.core import anchor_filter
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.core import constants
from avod.core import losses
from avod.core import model
from avod.core import summary_utils
from avod.core.anchor_generators import grid_anchor_3d_generator
import avod.core.proposal2bbox as ppb
from avod.core.box_3d_encoder import box_3d_to_anchor, box_3d_to_avod_projected_in
from avod.datasets.kitti import kitti_aug

import sys
sys.path.append('/home/zhangy/yolov3')
from core.yolov3_z import YOLOV3
from core.yolov3_z import DARKNET
# import core.common as common
import core.utils as utils

sys.path.append('/home/zhangy/avod/wavedata')
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils


class RpnModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    FEED_IS_TRAINING = 'feed_is_traning'
    PL_BEV_INPUT = 'bev_input_pl'
    PL_IMG_INPUT = 'img_input_pl'
    PL_YOLO_INPUT = 'yolov3_input_pl'
    PL_IMG_SHAPE = 'img_shape_pl'
    # PL_ANCHORS = 'anchors_pl'
    PL_GEN_BOX = 'top_gen_box'
    PL_TOP_ANCHORS = 'top_anchors_pl'
    PL_LABEL_SBBOX = 'label_sbbox_pl'
    PL_LABEL_MBBOX = 'label_mbbox_pl'
    PL_LABEL_LBBOX = 'label_lbbox_pl'
    PL_SBBOXES = 'sbboxes_pl'
    PL_MBBOXES = 'mbboxes_pl'
    PL_LBBOXES = 'lbboxes_pl'

    # PL_BEV_ANCHORS = 'bev_anchors_pl'
    # PL_BEV_ANCHORS_NORM = 'bev_anchors_norm_pl'
    # PL_IMG_ANCHORS = 'img_anchors_pl'
    # PL_IMG_ANCHORS_NORM = 'img_anchors_norm_pl'
    PL_LABEL_ANCHORS = 'label_anchors_pl'
    PL_LABEL_BOXES_3D = 'label_boxes_3d_pl'
    PL_LABEL_CLASSES = 'label_classes_pl'

    # PL_ANCHOR_IOUS = 'anchor_ious_pl'
    # PL_ANCHOR_OFFSETS = 'anchor_offsets_pl'
    # PL_ANCHOR_CLASSES = 'anchor_classes_pl'

    # Sample info, including keys for projection to image space
    # (e.g. camera matrix, image index, etc.)
    PL_CALIB_P2 = 'frame_calib_p2'
    PL_CALIB_P2_I = 'frame_calib_p2_I'
    PL_IMG_IDX = 'current_img_idx'
    PL_GROUND_PLANE = 'ground_plane'

    ##############################
    # Keys for Predictions
    ##############################
    PRED_ANCHORS = 'rpn_anchors'

    PRED_MB_OBJECTNESS_GT = 'rpn_mb_objectness_gt'
    PRED_MB_OFFSETS_GT = 'rpn_mb_offsets_gt'

    PRED_MB_MASK = 'rpn_mb_mask'
    PRED_MB_OBJECTNESS = 'rpn_mb_objectness'
    PRED_MB_OFFSETS = 'rpn_mb_offsets'

    PRED_TOP_INDICES = 'rpn_top_indices'
    PRED_TOP_ANCHORS = 'rpn_top_anchors'
    PRED_PRPS = 'yolo_prps'
    PRED_TOP_OBJECTNESS_SOFTMAX = 'rpn_top_objectness_softmax'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_RPN_OBJECTNESS = 'rpn_objectness_loss'
    LOSS_RPN_REGRESSION = 'rpn_regression_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(RpnModel, self).__init__(model_config)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test

        self._is_training = (self._train_val_test == 'train')

        # Input config
        input_config = self._config.input_config
        # self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
        #                                    input_config.bev_dims_w])
        self._bev_depth = input_config.bev_depth

        # self._img_pixel_size = np.asarray([input_config.img_dims_h,
        #                                    input_config.img_dims_w])
        self._img_depth = input_config.img_depth

        # Rpn config
        rpn_config = self._config.rpn_config
        self._proposal_roi_crop_size = \
            [rpn_config.rpn_proposal_roi_crop_size] * 2
        self._fusion_method = rpn_config.rpn_fusion_method

        if self._train_val_test in ["train", "val"]:
            self._nms_size = rpn_config.rpn_train_nms_size
        else:
            self._nms_size = rpn_config.rpn_test_nms_size

        self._nms_iou_thresh = rpn_config.rpn_nms_iou_thresh

        # Feature Extractor Nets
        # self._bev_feature_extractor = \
        #     feature_extractor_builder.get_extractor(
        #         self._config.layers_config.bev_feature_extractor)
        # self._img_feature_extractor = \
        #     feature_extractor_builder.get_extractor(
        #         self._config.layers_config.img_feature_extractor)

        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        # Information about the current sample
        self.sample_info = dict()

        self.write_2D_file = False  # (self._train_val_test != 'train')
        self.show_2D_bbox = False
        self.show_bbox_in_bev = False
        # self.batch_size = 1
        # Dataset
        self.dataset = dataset
        self.dataset.train_val_test = self._train_val_test
        self._area_extents = self.dataset.kitti_utils.area_extents
        self._bev_extents = self.dataset.kitti_utils.bev_extents
        self._cluster_sizes, _ = self.dataset.get_cluster_info()
        self._anchor_strides = self.dataset.kitti_utils.anchor_strides
        self._anchor_generator = \
            grid_anchor_3d_generator.GridAnchor3dGenerator()

        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._train_on_all_samples = self._config.train_on_all_samples
        self._eval_all_samples = self._config.eval_all_samples
        # Overwrite the dataset's variable with the config
        self.dataset.train_on_all_samples = self._train_on_all_samples

        # yolo v3 config
        self.max_bbox_per_scale = 150
        self.anchor_per_scale = 3
        self.num_classes = self.dataset.num_classes
        self.strides = np.array([8, 16, 32])
        self.yolo_model = YOLOV3
        self.darknet_model = DARKNET
        self.upsample_method = "resize"
        self.score_threshold = 0.1  # 0.3
        self.iou_threshold = 0.45
        self.train_input_size = 416  # 384

        if self._train_val_test in ["val", "test"]:
            # Disable path-drop, this should already be disabled inside the
            # evaluator, but just in case.
            self._path_drop_probabilities[0] = 1.0
            self._path_drop_probabilities[1] = 1.0

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        # 11train batch size=1:
        with tf.variable_scope('pl_labels'):
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_LABEL_ANCHORS)
            self._add_placeholder(tf.float32, [None, 7],
                                  self.PL_LABEL_BOXES_3D)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_LABEL_CLASSES)
        # 22train batch size>1:
        # for i in range(self.batch_size):
        #     with tf.variable_scope('pl_labels'):
        #         self._add_placeholder(tf.float32, [None, 6],
        #                               self.PL_LABEL_ANCHORS + str(i))
        #         self._add_placeholder(tf.float32, [None, 7],
        #                               self.PL_LABEL_BOXES_3D + str(i))
        #         self._add_placeholder(tf.float32, [None],
        #                               self.PL_LABEL_CLASSES + str(i))

        # Placeholders for anchors
        with tf.variable_scope('pl_anchors'):
            # self._add_placeholder(tf.float32, [None, 6],
            #                       self.PL_ANCHORS)
            # self._add_placeholder(tf.float32, [None],
            #                       self.PL_ANCHOR_IOUS)
            # self._add_placeholder(tf.float32, [None, 6],
            #                       self.PL_ANCHOR_OFFSETS)
            # self._add_placeholder(tf.float32, [None],
            #                       self.PL_ANCHOR_CLASSES)
            #
            # with tf.variable_scope('bev_anchor_projections'):
            #     self._add_placeholder(tf.float32, [None, 4],
            #                           self.PL_BEV_ANCHORS)
            #     self._bev_anchors_norm_pl = self._add_placeholder(
            #         tf.float32, [None, 4], self.PL_BEV_ANCHORS_NORM)
            #
            # with tf.variable_scope('img_anchor_projections'):
            #     self._add_placeholder(tf.float32, [None, 4],
            #                           self.PL_IMG_ANCHORS)
            #     self._img_anchors_norm_pl = self._add_placeholder(
            #         tf.float32, [None, 4], self.PL_IMG_ANCHORS_NORM)

            with tf.variable_scope('sample_info'):
                # the calib matrix shape is (3 x 4)
                self._add_placeholder(
                    tf.float32, [3, 4], self.PL_CALIB_P2)
                p2_I = self._add_placeholder(tf.float32, [4, 3], self.PL_CALIB_P2_I)
                self.p2_I = tf.expand_dims(p2_I, axis=0)
                self._add_placeholder(tf.int32,
                                      shape=[1],
                                      name=self.PL_IMG_IDX)
                self._add_placeholder(tf.float32, [4], self.PL_GROUND_PLANE)
                # before resize, the shape of original image
                self._add_placeholder(tf.float32, [2], self.PL_IMG_SHAPE)

        # Combine config data
        # bev_dims = np.append(self._bev_pixel_size, self._bev_depth)
        bev_dims = [None, None, self._bev_depth]

        with tf.variable_scope('bev_input'):
            # Placeholder for BEV image input, to be filled in with feed_dict
            bev_input_placeholder = self._add_placeholder(tf.float32, bev_dims,
                                                          self.PL_BEV_INPUT)

            self._bev_input_batches = tf.expand_dims(
                bev_input_placeholder, axis=0)
            # self._bev_input_batches = bev_input_placeholder

            # self._bev_preprocessed = \
            #     self._bev_feature_extractor.preprocess_input(
            #         self._bev_input_batches, self._bev_pixel_size)

            # Summary Images
            # bev_summary_images = tf.split(
            #     self._bev_preprocessed, self._bev_depth, axis=-1)
            # tf.summary.image("bev_maps", bev_summary_images[-1],
            #                  max_outputs=self._bev_depth)

        # with tf.variable_scope('img_input'):
        #     # Take variable size input images
        #     img_input_placeholder = self._add_placeholder(
        #         tf.float32,
        #         [None, None, self._img_depth],
        #         self.PL_IMG_INPUT)
        #
        #     self._img_input_batches = tf.expand_dims(
        #         img_input_placeholder, axis=0)
        #
        #     self._img_preprocessed = \
        #         self._img_feature_extractor.preprocess_input(
        #             self._img_input_batches, self._img_pixel_size)
        #
        #     # Summary Image
        #     tf.summary.image("rgb_image", self._img_preprocessed,
        #                      max_outputs=2)

        # self._img_pixel_size = np.asarray([self.train_input_size, 3 * self.train_input_size])

        # yolov3 image输入
        with tf.variable_scope('yolov3_input'):
            # Take variable size input images    #[none,none,3]
            yolov3_input_placeholder = self._add_placeholder(
                tf.float32,
                [None, None, self._img_depth],
                self.PL_YOLO_INPUT)  # [None, None, self._img_depth]->[None, None, 3]

            self._yolov3_input_batches = tf.expand_dims(
                yolov3_input_placeholder, axis=0)  # 维度增加一维->[1, None, None, 3]
            # self._yolov3_input_batches = yolov3_input_placeholder

            # Summary Image
            # img_summary_images = tf.split(
            #     self._yolov3_input_batches, [3, 1], axis=-1)
            # tf.summary.image("rgb_image", img_summary_images[0],
            #                  max_outputs=self.batch_size)  # yolov3_input_placeholder

            # yolov3的输入，2D boxes
            # xi,yi,xa,ya,class_id,alpha
            # label_sbbox_placeholder = self._add_placeholder(tf.float32,
            #                                                 [None, None,
            #                                                  3, 6 + self.num_classes], self.PL_LABEL_SBBOX)
            # self.label_sbbox = tf.expand_dims(label_sbbox_placeholder, axis=0)
            # # self.label_sbbox = label_sbbox_placeholder
            # label_mbbox_placeholder = self._add_placeholder(tf.float32,
            #                                                 [None, None,
            #                                                  3, 6 + self.num_classes], self.PL_LABEL_MBBOX)
            # self.label_mbbox = tf.expand_dims(label_mbbox_placeholder, axis=0)
            # # self.label_mbbox = label_mbbox_placeholder
            # label_lbbox_placeholder = self._add_placeholder(tf.float32,
            #                                                 [None, None,
            #                                                  3, 6 + self.num_classes], self.PL_LABEL_LBBOX)
            # self.label_lbbox = tf.expand_dims(label_lbbox_placeholder, axis=0)
            # # self.label_lbbox = label_lbbox_placeholder
            #
            # true_sbboxes = self._add_placeholder(tf.float32, [self.max_bbox_per_scale, 4],
            #                                      self.PL_SBBOXES)
            # self.true_sbboxes = tf.expand_dims(true_sbboxes, axis=0)
            # # self.true_sbboxes = true_sbboxes
            # true_mbboxes = self._add_placeholder(tf.float32, [self.max_bbox_per_scale, 4],
            #                                      self.PL_MBBOXES)
            # self.true_mbboxes = tf.expand_dims(true_mbboxes, axis=0)
            # # self.true_mbboxes = true_mbboxes
            # true_lbboxes = self._add_placeholder(tf.float32, [self.max_bbox_per_scale, 4],
            #                                      self.PL_LBBOXES)
            # self.true_lbboxes = tf.expand_dims(true_lbboxes, axis=0)
            # # self.true_lbboxes = true_lbboxes

        with tf.variable_scope('pl_top_anchors'):
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_TOP_ANCHORS)
            # self._add_placeholder(tf.float32, [None, 7],
            #                       self.PL_GEN_BOX)

    def _set_up_feature_extractors(self):
        """Sets up feature extractors and stores feature maps and
        bottlenecks as member variables.
        """

        # self.bev_feature_maps, self.bev_end_points = \
        #     self._bev_feature_extractor.build(
        #         self._bev_preprocessed,
        #         self._bev_pixel_size,
        #         self._is_training)

        # self.img_feature_maps, self.img_end_points = \
        #     self._img_feature_extractor.build(
        #         self._img_preprocessed,
        #         self._img_pixel_size,
        #         self._is_training)

        # yolo v3: Extract img features and pred 2D box
        self.model_img = self.yolo_model(self._yolov3_input_batches, self._is_training, self.p2_I, avod=True)
        # yolo v3+upsample*3->featuremap
        self.img_feature_maps = self.model_img.feature_maps
        # self.img_global_fe = self.model_img.darknet_fe

        # yolo v3: Extract bev features
        with tf.variable_scope('Extract_bev_features'):
            self.model_bev = self.darknet_model(self._bev_input_batches, self._is_training)
            self.bev_feature_maps = self.model_bev.feature_maps

        '''
        with tf.variable_scope('bev_bottleneck'):
            self.bev_bottleneck = slim.conv2d(
                self.bev_feature_maps,
                1, [1, 1],
                scope='bottleneck',
                normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'is_training': self._is_training})

        with tf.variable_scope('img_bottleneck'):
            self.img_bottleneck = slim.conv2d(
                self.img_feature_maps,
                1, [1, 1],
                scope='bottleneck',
                normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'is_training': self._is_training})
        '''

        # # Visualize the end point feature maps being used
        # for feature_map in list(self.bev_end_points.items()):
        #     if 'conv' in feature_map[0]:
        #         summary_utils.add_feature_maps_from_dict(self.bev_end_points,
        #                                                  feature_map[0])
        #
        # for feature_map in list(self.img_end_points.items()):
        #     if 'conv' in feature_map[0]:
        #         summary_utils.add_feature_maps_from_dict(self.img_end_points,
        #                                                  feature_map[0])

    def build(self):

        # Setup input placeholders
        self._set_up_input_pls()

        # Setup feature extractors
        self._set_up_feature_extractors()
        # top_anchors = self.placeholders[self.PL_TOP_ANCHORS]
        # img_shape = self.placeholders[self.PL_IMG_SHAPE]
        # stereo_calib_p2 = self.placeholders[self.PL_CALIB_P2]
        # sample_name = self.placeholders[self.PL_IMG_IDX]
        # pred_sbbox, pred_mbbox, pred_lbbox = self.model_img.pred_sbbox, self.model_img.pred_mbbox, self.model_img.pred_lbbox
        #
        # def process(pred_bbox, img_shape, stereo_calib_p2, sample_name):
        #     # boxes_process
        #     # print('sample_name', sample_name)
        #     # print('pred_bbox', pred_bbox)
        #     bboxesp = utils.postprocess_boxes_inavod(pred_bbox, img_shape, self.train_input_size, self.score_threshold)
        #     bboxes = utils.nms(bboxesp, self.iou_threshold)
        #     # print('bboxes', bboxes)
        #     proposal = [v[0:4] for v in bboxes]
        #     classid = [v[5] for v in bboxes]
        #     z = [v[6] for v in bboxes]
        #     p2 = np.array(stereo_calib_p2)
        #     # print('trainable', self._is_training)
        #     # print(self.batch_size)
        #
        #     if self.write_2D_file:
        #         classes = list(np.copy(self.dataset.classes))
        #         classes.extend(['DontCare'])
        #         predicted_dir_path = '/home/zhangy/yolov3/mAP/predicted'
        #         ground_truth_dir_path = '/home/zhangy/yolov3/mAP/ground-truth'
        #         predict_result_path = os.path.join(predicted_dir_path, str(sample_name) + '.txt')
        #         ground_truth_path = os.path.join(ground_truth_dir_path, str(sample_name) + '.txt')
        #
        #         if os.path.exists(predict_result_path):
        #             os.remove(predict_result_path)
        #         with open(predict_result_path, 'w') as f:
        #             for bbox in bboxesp:
        #                 coor = np.array(bbox[:4], dtype=np.int32)
        #                 score = bbox[4]
        #                 class_ind = int(bbox[5])
        #                 class_name = classes[class_ind]
        #                 score = '%.4f' % score
        #                 xmin, ymin, xmax, ymax = list(map(str, coor))
        #                 alpha = str(bbox[6])
        #                 bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax, alpha]) + '\n'
        #                 f.write(bbox_mess)
        #
        #         if os.path.exists(ground_truth_path) is False:
        #             annotation_path = os.path.join('/media/dataset/Kitti/object/training/label_2', '%06d.txt' % sample_name)
        #             with open(annotation_path, 'r') as file:
        #                 boxes = file.readlines()
        #             with open(ground_truth_path, 'w') as f:
        #                 for box in boxes:
        #                     class_name = str(box.split()[0])
        #                     if class_name in classes:
        #                         box2 = list(map(float, box.split()[4:8]))
        #                         xmin, ymin, xmax, ymax = list(map(str, box2))
        #                         alpha = str(box.split()[3])
        #                         bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax, alpha]) + '\n'
        #                         f.write(bbox_mess)
        #
        #         if self.show_2D_bbox:
        #             image_path = os.path.join('/media/dataset/Kitti/object/training/image_2', '00%04d.png' % sample_name)
        #             print(image_path)
        #             original_image = cv2.imread(image_path)
        #             original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        #             image = utils.draw_bbox(original_image, bboxes, classes=classes)
        #             image = Image.fromarray(image)
        #             image.show()
        #
        #             with open(ground_truth_path, 'r') as gt_file:
        #                 bboxes_gt = gt_file.readlines()
        #             proposalgt = []
        #             classgt = []
        #             if len(bboxes_gt) != 0:
        #                 for box in bboxes_gt:
        #                     box2 = list(map(float, box.split()[1:5]))
        #                     proposalgt.append(box2)
        #                     classidgt = classes.index(box.split()[0])
        #                     classgt.append(str(classidgt))
        #             image_gt = utils.draw2dbox(original_image, proposalgt, classgt, classes=classes,
        #                                        colors=[(0, 255, 0), (255, 255, 255)])
        #             image_gt = Image.fromarray(image_gt)
        #             image_gt.show()
        #
        #     # top_anchors
        #     avod_class = self.dataset.classes
        #     box_obj, box3, _ = ppb.GS3D().proposal2bbox(proposal, classid, p2, sample_name, z=z,
        #                                                 filter_class=avod_class)  # , alpha=alpha2d, aug=['multiple']
        #
        #     top_anchor = box_3d_to_anchor(box3)
        #     top_anchor = top_anchor.astype(np.float32)
        #     bboxes = np.array(bboxes, dtype=np.float32)
        #     return top_anchor, bboxes
        #
        # # 11train batch size=1:
        # pred_bbox = tf.concat([tf.reshape(pred_sbbox, (-1, 6 + self.num_classes)),
        #                        tf.reshape(pred_mbbox, (-1, 6 + self.num_classes)),
        #                        tf.reshape(pred_lbbox, (-1, 6 + self.num_classes))], axis=0)
        # top_anchors, prps = tf.py_func(process, [pred_bbox, img_shape, stereo_calib_p2, sample_name[0]],
        #                                [tf.float32, tf.float32])
        # top_anchors = tf.reshape(top_anchors, [-1, 6])
        # 22train batch size>1:
        # top_anchors = dict()
        # for i in range(self.batch_size):
        #     pred_bbox = tf.concat([tf.reshape(pred_sbbox[i], (-1, 6 + self.num_classes)),
        #                            tf.reshape(pred_mbbox[i], (-1, 6 + self.num_classes)),
        #                            tf.reshape(pred_lbbox[i], (-1, 6 + self.num_classes))], axis=0)
        #     top_anchor = tf.reshape(tf.py_func(process, [pred_bbox, img_shape[i], stereo_calib_p2[i],
        #                                                  sample_name[i][0]], tf.float32), [-1, 6])
        #     top_anchors['the %d batch top anchors' % i] = top_anchor

        # bev_proposal_input = self.bev_bottleneck
        # img_proposal_input = self.img_bottleneck

        # fusion_mean_div_factor = 2.0

        # If both img and bev probabilites are set to 1.0, don't do path drop.
        if not (self._path_drop_probabilities[0] ==
                self._path_drop_probabilities[1] == 1.0):
            with tf.variable_scope('rpn_path_drop'):

                random_values = tf.random_uniform(shape=[3],
                                                  minval=0.0,
                                                  maxval=1.0)

                img_mask, bev_mask = self.create_path_drop_masks(
                    self._path_drop_probabilities[0],
                    self._path_drop_probabilities[1],
                    random_values)

                # img_proposal_input = tf.multiply(img_proposal_input,
                #                                  img_mask)
                #
                # bev_proposal_input = tf.multiply(bev_proposal_input,
                #                                  bev_mask)

                self.img_path_drop_mask = img_mask
                self.bev_path_drop_mask = bev_mask

                # Overwrite the division factor
                # fusion_mean_div_factor = img_mask + bev_mask

        '''
        with tf.variable_scope('proposal_roi_pooling'):

            with tf.variable_scope('box_indices'):
                def get_box_indices(boxes):
                    proposals_shape = boxes.get_shape().as_list()
                    if any(dim is None for dim in proposals_shape):
                        proposals_shape = tf.shape(boxes)
                    ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                    multiplier = tf.expand_dims(
                        tf.range(start=0, limit=proposals_shape[0]), 1)
                    return tf.reshape(ones_mat * multiplier, [-1])

                bev_boxes_norm_batches = tf.expand_dims(
                    self._bev_anchors_norm_pl, axis=0)

                # These should be all 0's since there is only 1 image
                tf_box_indices = get_box_indices(bev_boxes_norm_batches)

            # Do ROI Pooling on BEV
            bev_proposal_rois = tf.image.crop_and_resize(
                bev_proposal_input,
                self._bev_anchors_norm_pl,
                tf_box_indices,
                self._proposal_roi_crop_size)
            # Do ROI Pooling on image
            img_proposal_rois = tf.image.crop_and_resize(
                img_proposal_input,
                self._img_anchors_norm_pl,
                tf_box_indices,
                self._proposal_roi_crop_size)

        with tf.variable_scope('proposal_roi_fusion'):
            rpn_fusion_out = None
            if self._fusion_method == 'mean':
                tf_features_sum = tf.add(bev_proposal_rois, img_proposal_rois)
                rpn_fusion_out = tf.divide(tf_features_sum,
                                           fusion_mean_div_factor)
            elif self._fusion_method == 'concat':
                rpn_fusion_out = tf.concat(
                    [bev_proposal_rois, img_proposal_rois], axis=3)
            else:
                raise ValueError('Invalid fusion method', self._fusion_method)

        # TODO: move this section into an separate AnchorPredictor class
        with tf.variable_scope('anchor_predictor', 'ap', [rpn_fusion_out]):
            tensor_in = rpn_fusion_out

            # Parse rpn layers config
            layers_config = self._config.layers_config.rpn_config
            l2_weight_decay = layers_config.l2_weight_decay

            if l2_weight_decay > 0:
                weights_regularizer = slim.l2_regularizer(l2_weight_decay)
            else:
                weights_regularizer = None

            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=weights_regularizer):
                # Use conv2d instead of fully_connected layers.
                cls_fc6 = slim.conv2d(tensor_in,
                                      layers_config.cls_fc6,
                                      self._proposal_roi_crop_size,
                                      padding='VALID',
                                      scope='cls_fc6')

                cls_fc6_drop = slim.dropout(cls_fc6,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='cls_fc6_drop')

                cls_fc7 = slim.conv2d(cls_fc6_drop,
                                      layers_config.cls_fc7,
                                      [1, 1],
                                      scope='cls_fc7')

                cls_fc7_drop = slim.dropout(cls_fc7,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='cls_fc7_drop')

                cls_fc8 = slim.conv2d(cls_fc7_drop,
                                      2,
                                      [1, 1],
                                      activation_fn=None,
                                      scope='cls_fc8')

                objectness = tf.squeeze(
                    cls_fc8, [1, 2],
                    name='cls_fc8/squeezed')

                # Use conv2d instead of fully_connected layers.
                reg_fc6 = slim.conv2d(tensor_in,
                                      layers_config.reg_fc6,
                                      self._proposal_roi_crop_size,
                                      padding='VALID',
                                      scope='reg_fc6')

                reg_fc6_drop = slim.dropout(reg_fc6,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='reg_fc6_drop')

                reg_fc7 = slim.conv2d(reg_fc6_drop,
                                      layers_config.reg_fc7,
                                      [1, 1],
                                      scope='reg_fc7')

                reg_fc7_drop = slim.dropout(reg_fc7,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='reg_fc7_drop')

                reg_fc8 = slim.conv2d(reg_fc7_drop,
                                      6,
                                      [1, 1],
                                      activation_fn=None,
                                      scope='reg_fc8')

                offsets = tf.squeeze(
                    reg_fc8, [1, 2],
                    name='reg_fc8/squeezed')

        # Histogram summaries
        with tf.variable_scope('histograms_feature_extractor'):
            with tf.variable_scope('bev_vgg'):
                for end_point in self.bev_end_points:
                    tf.summary.histogram(
                        end_point, self.bev_end_points[end_point])

            with tf.variable_scope('img_vgg'):
                for end_point in self.img_end_points:
                    tf.summary.histogram(
                        end_point, self.img_end_points[end_point])

        with tf.variable_scope('histograms_rpn'):
            with tf.variable_scope('anchor_predictor'):
                fc_layers = [cls_fc6, cls_fc7, cls_fc8, objectness,
                             reg_fc6, reg_fc7, reg_fc8, offsets]
                for fc_layer in fc_layers:
                    # fix the name to avoid tf warnings
                    tf.summary.histogram(fc_layer.name.replace(':', '_'),
                                         fc_layer)

        # Return the proposals
        with tf.variable_scope('proposals'):
            anchors = self.placeholders[self.PL_ANCHORS]

            # Decode anchor regression offsets
            with tf.variable_scope('decoding'):
                regressed_anchors = anchor_encoder.offset_to_anchor(
                        anchors, offsets)

            with tf.variable_scope('bev_projection'):
                _, bev_proposal_boxes_norm = anchor_projector.project_to_bev(
                    regressed_anchors, self._bev_extents)

            with tf.variable_scope('softmax'):
                objectness_softmax = tf.nn.softmax(objectness)

            with tf.variable_scope('nms'):
                objectness_scores = objectness_softmax[:, 1]

                # Do NMS on regressed anchors
                top_indices = tf.image.non_max_suppression(
                    bev_proposal_boxes_norm, objectness_scores,
                    max_output_size=self._nms_size,
                    iou_threshold=self._nms_iou_thresh)

                top_anchors = tf.gather(regressed_anchors, top_indices)
                top_objectness_softmax = tf.gather(objectness_scores,
                                                   top_indices)
                # top_offsets = tf.gather(offsets, top_indices)
                # top_objectness = tf.gather(objectness, top_indices)

        # Get mini batch
        all_ious_gt = self.placeholders[self.PL_ANCHOR_IOUS]
        all_offsets_gt = self.placeholders[self.PL_ANCHOR_OFFSETS]
        all_classes_gt = self.placeholders[self.PL_ANCHOR_CLASSES]

        with tf.variable_scope('mini_batch'):
            mini_batch_utils = self.dataset.kitti_utils.mini_batch_utils
            mini_batch_mask, _ = \
                mini_batch_utils.sample_rpn_mini_batch(all_ious_gt)

        # ROI summary images
        rpn_mini_batch_size = \
            self.dataset.kitti_utils.mini_batch_utils.rpn_mini_batch_size
        with tf.variable_scope('bev_rpn_rois'):
            mb_bev_anchors_norm = tf.boolean_mask(self._bev_anchors_norm_pl,
                                                  mini_batch_mask)
            mb_bev_box_indices = tf.zeros_like(
                tf.boolean_mask(all_classes_gt, mini_batch_mask),
                dtype=tf.int32)

            # Show the ROIs of the BEV input density map
            # for the mini batch anchors
            bev_input_rois = tf.image.crop_and_resize(
                self._bev_preprocessed,
                mb_bev_anchors_norm,
                mb_bev_box_indices,
                (32, 32))

            bev_input_roi_summary_images = tf.split(
                bev_input_rois, self._bev_depth, axis=3)
            tf.summary.image('bev_rpn_rois',
                             bev_input_roi_summary_images[-1],
                             max_outputs=rpn_mini_batch_size)

        with tf.variable_scope('img_rpn_rois'):
            # ROIs on image input
            mb_img_anchors_norm = tf.boolean_mask(self._img_anchors_norm_pl,
                                                  mini_batch_mask)
            mb_img_box_indices = tf.zeros_like(
                tf.boolean_mask(all_classes_gt, mini_batch_mask),
                dtype=tf.int32)

            # Do test ROI pooling on mini batch
            img_input_rois = tf.image.crop_and_resize(
                self._img_preprocessed,
                mb_img_anchors_norm,
                mb_img_box_indices,
                (32, 32))

            tf.summary.image('img_rpn_rois',
                             img_input_rois,
                             max_outputs=rpn_mini_batch_size)

        # Ground Truth Tensors
        with tf.variable_scope('one_hot_classes'):

            # Anchor classification ground truth
            # Object / Not Object
            min_pos_iou = \
                self.dataset.kitti_utils.mini_batch_utils.rpn_pos_iou_range[0]

            objectness_classes_gt = tf.cast(
                tf.greater_equal(all_ious_gt, min_pos_iou),
                dtype=tf.int32)
            objectness_gt = tf.one_hot(
                objectness_classes_gt, depth=2,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=self._config.label_smoothing_epsilon)

        # Mask predictions for mini batch
        with tf.variable_scope('prediction_mini_batch'):
            objectness_masked = tf.boolean_mask(objectness, mini_batch_mask)
            offsets_masked = tf.boolean_mask(offsets, mini_batch_mask)

        with tf.variable_scope('ground_truth_mini_batch'):
            objectness_gt_masked = tf.boolean_mask(
                objectness_gt, mini_batch_mask)
            offsets_gt_masked = tf.boolean_mask(all_offsets_gt,
                                                mini_batch_mask)
        '''

        # Specify the tensors to evaluate
        predictions = dict()

        # Temporary predictions for debugging
        # predictions['anchor_ious'] = anchor_ious
        # predictions['anchor_offsets'] = all_offsets_gt

        # if self._train_val_test in ['train', 'val']:
        #     # All anchors
        #     predictions[self.PRED_ANCHORS] = anchors
        #
        #     # Mini-batch masks
        #     predictions[self.PRED_MB_MASK] = mini_batch_mask
        #     # Mini-batch predictions
        #     predictions[self.PRED_MB_OBJECTNESS] = objectness_masked
        #     predictions[self.PRED_MB_OFFSETS] = offsets_masked
        #
        #     # Mini batch ground truth
        #     predictions[self.PRED_MB_OFFSETS_GT] = offsets_gt_masked
        #     predictions[self.PRED_MB_OBJECTNESS_GT] = objectness_gt_masked
        #
        #     # Proposals after nms
        #     predictions[self.PRED_TOP_INDICES] = top_indices
        #     predictions[self.PRED_TOP_ANCHORS] = top_anchors
        #     predictions[
        #         self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax
        #
        # else:
        #     # self._train_val_test == 'test'
        #     predictions[self.PRED_TOP_ANCHORS] = top_anchors
        #     predictions[
        #         self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax

        # predictions[self.PRED_TOP_ANCHORS] = top_anchors
        # predictions[self.PRED_PRPS] = prps

        # predictions[self.PL_LABEL_SBBOX] = self.label_sbbox
        # predictions[self.PL_LABEL_MBBOX] = self.label_mbbox
        # predictions[self.PL_LABEL_LBBOX] = self.label_lbbox
        # predictions[self.PL_SBBOXES] = self.true_sbboxes
        # predictions[self.PL_MBBOXES] = self.true_mbboxes
        # predictions[self.PL_LBBOXES] = self.true_lbboxes

        return predictions

    def create_feed_dict(self, sample_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """

        # is_training = False
        # self.train_input_size = 384
        if self._train_val_test in ["train", "val"]:

            # sample_index should be None
            if sample_index is not None:
                raise ValueError('sample_index should be None. Do not load '
                                 'particular samples during train or val')

            # During training/validation, we need a valid sample
            # with anchor info for loss calculation
            sample = None
            anchors_info = []

            valid_sample = False
            while not valid_sample:
                if self._train_val_test == "train":
                    # Get the a random sample from the remaining epoch
                    samples = self.dataset.next_batch(batch_size=1)
                    # is_training = True
                    # self.train_input_sizes = [320, 352, 384]
                    # self.train_input_size = random.choice(self.train_input_sizes)

                else:  # self._train_val_test == "val"
                    # Load samples in order for validation
                    samples = self.dataset.next_batch(batch_size=1,
                                                      shuffle=False)

                # Only handle one sample at a time for now
                sample = samples[0]
                anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

                # When training, if the mini batch is empty, go to the next
                # sample. Otherwise carry on with found the valid sample.
                # For validation, even if 'anchors_info' is empty, keep the
                # sample (this will help penalize false positives.)
                # We will substitue the necessary info with zeros later on.
                # Note: Training/validating all samples can be switched off.
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)
                if (anchors_info == []) or train_cond or eval_cond:
                    valid_sample = True
        else:
            # For testing, any sample should work
            if sample_index is not None:
                samples = self.dataset.load_samples([sample_index])
            else:
                samples = self.dataset.next_batch(batch_size=1, shuffle=False)

            # Only handle one sample at a time for now
            sample = samples[0]
            anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

        sample_name = sample.get(constants.KEY_SAMPLE_NAME)
        sample_augs = sample.get(constants.KEY_SAMPLE_AUGS)

        # Get ground truth data
        label_anchors = sample.get(constants.KEY_LABEL_ANCHORS)
        label_classes = sample.get(constants.KEY_LABEL_CLASSES)
        # We only need orientation from box_3d
        label_boxes_3d = sample.get(constants.KEY_LABEL_BOXES_3D)

        # Network input data
        image_input = sample.get(constants.KEY_IMAGE_INPUT)
        bev_input = sample.get(constants.KEY_BEV_INPUT)

        # Image shape (h, w)
        image_shape = [image_input.shape[0], image_input.shape[1]]

        ground_plane = sample.get(constants.KEY_GROUND_PLANE)
        stereo_calib_p2 = sample.get(constants.KEY_STEREO_CALIB_P2)
        stereo_calib_p2_I = sample.get(constants.KEY_P2_I)

        # # Fill the placeholders for anchor information
        # self._fill_anchor_pl_inputs(anchors_info=anchors_info,
        #                             ground_plane=ground_plane,
        #                             image_shape=image_shape,
        #                             stereo_calib_p2=stereo_calib_p2,
        #                             sample_name=sample_name,
        #                             sample_augs=sample_augs)

        # Use the 3D box generated by gt 2D box as top anchors to run AVOD detection network
        avod_class = self.dataset.classes
        box_path = os.path.join('/home/zhangy/yolov3/mAP/predicted_trainzz_0.1/', str(int(sample_name)) + '.txt')
        with open(box_path, 'r') as file:
            boxes = file.readlines()
            proposal = []
            classes = []
            alpha = []
            z = []
            if len(boxes) != 0:
                for box in boxes:
                    classid = avod_class.index(box.split()[0])
                    if classid == 0:
                        box2 = list(map(float, box.split()[2:6]))
                        proposal.append(box2)
                        classes.append(str(classid))
                        # alpha.append(box.split()[7])
                        z.append(box.split()[6])
        _, box3, __ = ppb.GS3D().proposal2bbox(
            proposal, classes, stereo_calib_p2, sample_name, z=z, filter_class=avod_class)  # , alpha=alpha, aug=['multiple']
        top_anchors = box_3d_to_anchor(box3)
        # top_anchors = box_3d_to_avod_projected_in(box3)
        # self._placeholder_inputs[self.PL_GEN_BOX] = box3
        self._placeholder_inputs[self.PL_TOP_ANCHORS] = top_anchors

        # Input for yolov3
        bboxes = sample.get(constants.KEY_LABEL_BOXES_2D)  # （N, 6）
        image, bboxes = utils.image_preporcess(np.copy(image_input),
                                               [self.train_input_size, 3 * self.train_input_size], np.copy(bboxes))
        # label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

        bev = cv2.resize(np.copy(bev_input), (2 * self.train_input_size, 2 * self.train_input_size))
        # self._placeholder_inputs[self.PL_LABEL_SBBOX] = label_sbbox
        # self._placeholder_inputs[self.PL_LABEL_MBBOX] = label_mbbox
        # self._placeholder_inputs[self.PL_LABEL_LBBOX] = label_lbbox
        # self._placeholder_inputs[self.PL_SBBOXES] = sbboxes
        # self._placeholder_inputs[self.PL_MBBOXES] = mbboxes
        # self._placeholder_inputs[self.PL_LBBOXES] = lbboxes

        # this is a list to match the explicit shape for the placeholder
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]

        # Fill in the rest
        self._placeholder_inputs[self.PL_BEV_INPUT] = bev
        # self._placeholder_inputs[self.PL_IMG_INPUT] = image_input
        self._placeholder_inputs[self.PL_YOLO_INPUT] = image
        self._placeholder_inputs[self.PL_IMG_SHAPE] = image_shape
        # self._placeholder_inputs[self.FEED_IS_TRAINING] = is_training

        self._placeholder_inputs[self.PL_LABEL_ANCHORS] = label_anchors
        self._placeholder_inputs[self.PL_LABEL_BOXES_3D] = label_boxes_3d
        self._placeholder_inputs[self.PL_LABEL_CLASSES] = label_classes

        # Sample Info
        # img_idx is a list to match the placeholder shape
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]
        self._placeholder_inputs[self.PL_CALIB_P2] = stereo_calib_p2
        self._placeholder_inputs[self.PL_CALIB_P2_I] = stereo_calib_p2_I
        self._placeholder_inputs[self.PL_GROUND_PLANE] = ground_plane

        # Temporary sample info for debugging
        self.sample_info.clear()
        self.sample_info['sample_name'] = sample_name
        # self.sample_info['rpn_mini_batch'] = anchors_info

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]

        return feed_dict

    '''
    def create_feed_dict(self, sample_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """
        self.batch_size = 1
        self.train_input_size = 384
        self.train_output_sizes = self.train_input_size // self.strides
        is_training = False
        if self._train_val_test == "train":
            self.batch_size = 4
            is_training = True
            self.train_input_sizes = [320, 352, 384]
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides

        batch_image = np.zeros((self.batch_size, self.train_input_size, 3 * self.train_input_size, 3))
        batch_label_sbbox = np.zeros(
            (self.batch_size, self.train_output_sizes[0], 3 * self.train_output_sizes[0],
             self.anchor_per_scale, 6 + self.num_classes))
        batch_label_mbbox = np.zeros(
            (self.batch_size, self.train_output_sizes[1], 3 * self.train_output_sizes[1],
             self.anchor_per_scale, 6 + self.num_classes))
        batch_label_lbbox = np.zeros(
            (self.batch_size, self.train_output_sizes[2], 3 * self.train_output_sizes[2],
             self.anchor_per_scale, 6 + self.num_classes))
        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_bev_input = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 6))
        batch_sample_name = np.zeros((self.batch_size, 1))
        batch_image_shape = np.zeros((self.batch_size, 2))
        batch_stereo_calib_p2 = np.zeros((self.batch_size, 3, 4))
        batch_ground_plane = np.zeros((self.batch_size, 4))

        samples = None
        if self._train_val_test in ["train", "val"]:

            # sample_index should be None
            if sample_index is not None:
                raise ValueError('sample_index should be None. Do not load '
                                 'particular samples during train or val')

            valid_sample = False
            while not valid_sample:
                if self._train_val_test == "train":
                    # Get the a random sample from the remaining epoch
                    samples = self.dataset.next_batch(batch_size=self.batch_size)

                    if len(samples) != self.batch_size:
                        continue
                    valid_sample = True

                else:  # self._train_val_test == "val"
                    # Load samples in order for validation
                    samples = self.dataset.next_batch(batch_size=self.batch_size,
                                                      shuffle=False)
                    if len(samples) != 1:
                        continue
                    valid_sample = True

        else:
            # For testing, any sample should work
            if sample_index is not None:
                samples = self.dataset.load_samples([sample_index])
            else:
                samples = self.dataset.next_batch(batch_size=self.batch_size, shuffle=False)

        for num in range(self.batch_size):
            sample = samples[num]

            # Network input data
            image_input = sample.get(constants.KEY_IMAGE_INPUT)  # (375, 1242, 3）：图片大小，数据仅此样本
            bev_input = sample.get(constants.KEY_BEV_INPUT)  # （700， 800， 6）：bev大小

            # 用于yolov3的输入
            bboxes = sample.get(constants.KEY_LABEL_BOXES_2D)  # （N, 6）
            image, bboxes = self.image_preporcess(np.copy(image_input),
                                                  [self.train_input_size, 3 * self.train_input_size],
                                                  np.copy(bboxes))
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                bboxes)
            bev = cv2.resize(np.copy(bev_input), (self.train_input_size, self.train_input_size))

            sample_name = sample.get(constants.KEY_SAMPLE_NAME)  # str：00xxxx

            # Get ground truth data
            label_anchors = sample.get(constants.KEY_LABEL_ANCHORS)  # （N, 6）
            label_classes = sample.get(constants.KEY_LABEL_CLASSES)  # （N,）
            # We only need orientation from box_3d
            label_boxes_3d = sample.get(constants.KEY_LABEL_BOXES_3D)  # （N, 7）

            # 用于avod model的gt
            self._placeholder_inputs[self.PL_LABEL_ANCHORS + str(num)] = label_anchors
            self._placeholder_inputs[self.PL_LABEL_BOXES_3D + str(num)] = label_boxes_3d
            self._placeholder_inputs[self.PL_LABEL_CLASSES + str(num)] = label_classes

            # Image shape (h, w)
            image_shape = [image_input.shape[0], image_input.shape[1]]

            ground_plane = sample.get(constants.KEY_GROUND_PLANE)  # （4，）
            stereo_calib_p2 = sample.get(constants.KEY_STEREO_CALIB_P2)  # （3, 4）

            batch_image[num, :, :, :] = image
            batch_label_sbbox[num, :, :, :, :] = label_sbbox
            batch_label_mbbox[num, :, :, :, :] = label_mbbox
            batch_label_lbbox[num, :, :, :, :] = label_lbbox
            batch_sbboxes[num, :, :] = sbboxes
            batch_mbboxes[num, :, :] = mbboxes
            batch_lbboxes[num, :, :] = lbboxes
            batch_bev_input[num, :, :, :] = bev
            batch_sample_name[num, :] = [int(sample_name)]
            batch_image_shape[num, :] = image_shape
            batch_stereo_calib_p2[num, :, :] = stereo_calib_p2
            batch_ground_plane[num, :] = ground_plane

        self._placeholder_inputs[self.PL_LABEL_SBBOX] = batch_label_sbbox
        self._placeholder_inputs[self.PL_LABEL_MBBOX] = batch_label_mbbox
        self._placeholder_inputs[self.PL_LABEL_LBBOX] = batch_label_lbbox
        self._placeholder_inputs[self.PL_SBBOXES] = batch_sbboxes
        self._placeholder_inputs[self.PL_MBBOXES] = batch_mbboxes
        self._placeholder_inputs[self.PL_LBBOXES] = batch_lbboxes

        # this is a list to match the explicit shape for the placeholder
        self._placeholder_inputs[self.PL_IMG_IDX] = batch_sample_name

        # Fill in the rest
        self._placeholder_inputs[self.PL_BEV_INPUT] = batch_bev_input
        # self._placeholder_inputs[self.PL_IMG_INPUT] = image_input
        self._placeholder_inputs[self.PL_YOLO_INPUT] = batch_image
        self._placeholder_inputs[self.PL_IMG_SHAPE] = batch_image_shape

        # Sample Info
        # img_idx is a list to match the placeholder shape
        # self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]
        self._placeholder_inputs[self.PL_CALIB_P2] = batch_stereo_calib_p2
        self._placeholder_inputs[self.PL_GROUND_PLANE] = batch_ground_plane

        self._placeholder_inputs[self.FEED_IS_TRAINING] = is_training

        # Temporary sample info for debugging
        self.sample_info.clear()
        # if self.batch_size == 1:
        self.sample_info['sample_name'] = batch_sample_name
        # self.sample_info['rpn_mini_batch'] = anchors_info

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]

        return feed_dict
    '''
    '''
    def create_feed_dict_test(self, dataset_test=None, sample_index=None):
        self.batch_size = 1
        self.train_input_size = 384
        self.train_output_sizes = self.train_input_size // self.strides

        batch_image = np.zeros((self.batch_size, self.train_input_size, 3 * self.train_input_size, 3))
        batch_label_sbbox = np.zeros(
            (self.batch_size, self.train_output_sizes[0], 3 * self.train_output_sizes[0],
             self.anchor_per_scale, 6 + self.num_classes))
        batch_label_mbbox = np.zeros(
            (self.batch_size, self.train_output_sizes[1], 3 * self.train_output_sizes[1],
             self.anchor_per_scale, 6 + self.num_classes))
        batch_label_lbbox = np.zeros(
            (self.batch_size, self.train_output_sizes[2], 3 * self.train_output_sizes[2],
             self.anchor_per_scale, 6 + self.num_classes))
        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_bev_input = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 6))
        batch_sample_name = np.zeros((self.batch_size, 1))
        batch_image_shape = np.zeros((self.batch_size, 2))
        batch_stereo_calib_p2 = np.zeros((self.batch_size, 3, 4))
        batch_ground_plane = np.zeros((self.batch_size, 4))

        samples = None
        if dataset_test is None:
            dataset_test = self.dataset
        if self._train_val_test in ["train", "val"]:

            # sample_index should be None
            if sample_index is not None:
                raise ValueError('sample_index should be None. Do not load '
                                 'particular samples during train or val')

            valid_sample = False
            while not valid_sample:
                samples = self.dataset.next_batch(batch_size=self.batch_size,
                                                  shuffle=False)
                if len(samples) != 1:
                    continue
                valid_sample = True
        else:
            # For testing, any sample should work
            if sample_index is not None:
                samples = dataset_test.load_samples([sample_index])
            else:
                samples = dataset_test.next_batch(batch_size=self.batch_size, shuffle=False)

        for num in range(self.batch_size):
            sample = samples[num]

            # Network input data
            image_input = sample.get(constants.KEY_IMAGE_INPUT)  # (375, 1242, 3）：图片大小，数据仅此样本
            bev_input = sample.get(constants.KEY_BEV_INPUT)  # （700， 800， 6）：bev大小

            # 用于yolov3的输入
            bboxes = sample.get(constants.KEY_LABEL_BOXES_2D)  # （N, 6）
            image, bboxes = self.image_preporcess(np.copy(image_input),
                                                  [self.train_input_size, 3 * self.train_input_size],
                                                  np.copy(bboxes))
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                bboxes)
            bev = self.image_preporcess(np.copy(bev_input), [self.train_input_size, self.train_input_size])

            sample_name = sample.get(constants.KEY_SAMPLE_NAME)  # str：00xxxx

            # Get ground truth data
            label_anchors = sample.get(constants.KEY_LABEL_ANCHORS)  # （N, 6）
            label_classes = sample.get(constants.KEY_LABEL_CLASSES)  # （N,）
            # We only need orientation from box_3d
            label_boxes_3d = sample.get(constants.KEY_LABEL_BOXES_3D)  # （N, 7）

            # 用于avod model的gt
            self._placeholder_inputs[self.PL_LABEL_ANCHORS + str(num)] = label_anchors
            self._placeholder_inputs[self.PL_LABEL_BOXES_3D + str(num)] = label_boxes_3d
            self._placeholder_inputs[self.PL_LABEL_CLASSES + str(num)] = label_classes

            # Image shape (h, w)
            image_shape = [image_input.shape[0], image_input.shape[1]]

            ground_plane = sample.get(constants.KEY_GROUND_PLANE)  # （4，）
            stereo_calib_p2 = sample.get(constants.KEY_STEREO_CALIB_P2)  # （3, 4）

            batch_image[num, :, :, :] = image
            batch_label_sbbox[num, :, :, :, :] = label_sbbox
            batch_label_mbbox[num, :, :, :, :] = label_mbbox
            batch_label_lbbox[num, :, :, :, :] = label_lbbox
            batch_sbboxes[num, :, :] = sbboxes
            batch_mbboxes[num, :, :] = mbboxes
            batch_lbboxes[num, :, :] = lbboxes
            batch_bev_input[num, :, :, :] = bev
            batch_sample_name[num, :] = [int(sample_name)]
            batch_image_shape[num, :] = image_shape
            batch_stereo_calib_p2[num, :, :] = stereo_calib_p2
            batch_ground_plane[num, :] = ground_plane

        self._placeholder_inputs[self.PL_LABEL_SBBOX] = batch_label_sbbox
        self._placeholder_inputs[self.PL_LABEL_MBBOX] = batch_label_mbbox
        self._placeholder_inputs[self.PL_LABEL_LBBOX] = batch_label_lbbox
        self._placeholder_inputs[self.PL_SBBOXES] = batch_sbboxes
        self._placeholder_inputs[self.PL_MBBOXES] = batch_mbboxes
        self._placeholder_inputs[self.PL_LBBOXES] = batch_lbboxes

        # this is a list to match the explicit shape for the placeholder
        self._placeholder_inputs[self.PL_IMG_IDX] = batch_sample_name

        # Fill in the rest
        self._placeholder_inputs[self.PL_BEV_INPUT] = batch_bev_input
        # self._placeholder_inputs[self.PL_IMG_INPUT] = image_input
        self._placeholder_inputs[self.PL_YOLO_INPUT] = batch_image
        self._placeholder_inputs[self.PL_IMG_SHAPE] = batch_image_shape

        # Sample Info
        # img_idx is a list to match the placeholder shape
        # self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]
        self._placeholder_inputs[self.PL_CALIB_P2] = batch_stereo_calib_p2
        self._placeholder_inputs[self.PL_GROUND_PLANE] = batch_ground_plane

        self._placeholder_inputs[self.FEED_IS_TRAINING] = False

        # Temporary sample info for debugging
        self.sample_info.clear()
        # if self.batch_size == 1:
        self.sample_info['sample_name'] = batch_sample_name
        # self.sample_info['rpn_mini_batch'] = anchors_info

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]

        return feed_dict
    '''

    def image_preporcess(self, image, target_size, gt_boxes=None):

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # resize+pad
        # ih, iw = target_size
        # h, w, _ = image.shape
        #
        # scale = min(iw / w, ih / h)
        # nw, nh = int(scale * w), int(scale * h)
        # image_resized = cv2.resize(image, (nw, nh))
        #
        # image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        # dw, dh = (iw - nw) // 2, (ih - nh) // 2
        # image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        # image_paded = image_paded / 255.
        #
        # if gt_boxes is None:
        #     return image_paded
        #
        # else:
        #     gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        #     gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        #     return image_paded, gt_boxes

        # only resize
        ih, iw = target_size
        h, w, _ = image.shape

        scale_w = iw / w
        scale_h = ih / h
        image_resized = cv2.resize(image, (iw, ih))
        image_resized[:, :, 0:3] = image_resized[:, :, 0:3] / 255.  # [:, :, 0:3]

        if gt_boxes is None:
            return image_resized

        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale_w
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale_h
            return image_resized, gt_boxes

    def preprocess_true_boxes(self, bboxes):

        anchors_path = "/home/zhangy/yolov3/data/anchors/basline_anchors.txt"
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = np.array(anchors.split(','), dtype=np.float32)

        self.train_output_sizes = self.train_input_size // self.strides
        label = [np.zeros((self.train_output_sizes[i], 3 * self.train_output_sizes[i], self.anchor_per_scale,
                           6 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            # bbox_alpha = bbox[5]
            bbox_z = bbox[5]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[int(bbox_class_ind)] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution  # (num_classes,)

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)  # (4,)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]  # (3, 4)

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))  # (3, 4)
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)  # (3,)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # from xi, yi, xa, ya, class_ind to x, y, w, h, conf, class
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = bbox_z
                    label[i][yind, xind, iou_mask, 6:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_z
                label[best_detect][yind, xind, best_anchor, 6:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        #  from x, y, w, h to xi, yi, xa, ya
        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def _fill_anchor_pl_inputs(self,
                               anchors_info,
                               ground_plane,
                               image_shape,
                               stereo_calib_p2,
                               sample_name,
                               sample_augs):
        """
        Fills anchor placeholder inputs with corresponding data

        Args:
            anchors_info: anchor info from mini_batch_utils
            ground_plane: ground plane coefficients
            image_shape: image shape (h, w), used for projecting anchors
            sample_name: name of the sample, e.g. "000001"
            sample_augs: list of sample augmentations
        """

        # Lists for merging anchors info
        all_anchor_boxes_3d = []
        anchors_ious = []
        anchor_offsets = []
        anchor_classes = []

        # Create anchors for each class
        if len(self.dataset.classes) > 1:
            for class_idx in range(len(self.dataset.classes)):
                # Generate anchors for all classes
                grid_anchor_boxes_3d = self._anchor_generator.generate(
                    area_3d=self._area_extents,
                    anchor_3d_sizes=self._cluster_sizes[class_idx],
                    anchor_stride=self._anchor_strides[class_idx],
                    ground_plane=ground_plane)
                all_anchor_boxes_3d.append(grid_anchor_boxes_3d)
            all_anchor_boxes_3d = np.concatenate(all_anchor_boxes_3d)
        else:
            # Don't loop for a single class
            class_idx = 0
            grid_anchor_boxes_3d = self._anchor_generator.generate(
                area_3d=self._area_extents,
                anchor_3d_sizes=self._cluster_sizes[class_idx],
                anchor_stride=self._anchor_strides[class_idx],
                ground_plane=ground_plane)  # 89600*7
            all_anchor_boxes_3d = grid_anchor_boxes_3d

        # Filter empty anchors
        # Skip if anchors_info is []
        sample_has_labels = True
        if self._train_val_test in ['train', 'val']:
            # Read in anchor info during training / validation
            if anchors_info:
                anchor_indices, anchors_ious, anchor_offsets, \
                    anchor_classes = anchors_info

                anchor_boxes_3d_to_use = all_anchor_boxes_3d[anchor_indices]
            else:
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)
                if train_cond or eval_cond:
                    sample_has_labels = False
        else:
            sample_has_labels = False

        if not sample_has_labels:
            # During testing, or validation with no anchor info, manually
            # filter empty anchors
            # TODO: share voxel_grid_2d with BEV generation if possible
            voxel_grid_2d = \
                self.dataset.kitti_utils.create_sliced_voxel_grid_2d(
                    sample_name, self.dataset.bev_source,
                    image_shape=image_shape)

            # Convert to anchors and filter
            anchors_to_use = box_3d_encoder.box_3d_to_anchor(
                all_anchor_boxes_3d)  # 89600*6
            empty_filter = anchor_filter.get_empty_anchor_filter_2d(
                anchors_to_use, voxel_grid_2d, density_threshold=1)  # 89600,

            anchor_boxes_3d_to_use = all_anchor_boxes_3d[empty_filter]  # 9910*7(Remove empty anchor)

        # Convert lists to ndarrays
        anchor_boxes_3d_to_use = np.asarray(anchor_boxes_3d_to_use)
        anchors_ious = np.asarray(anchors_ious)
        anchor_offsets = np.asarray(anchor_offsets)
        anchor_classes = np.asarray(anchor_classes)

        # Flip anchors and centroid x offsets for augmented samples
        if kitti_aug.AUG_FLIPPING in sample_augs:
            anchor_boxes_3d_to_use = kitti_aug.flip_boxes_3d(
                anchor_boxes_3d_to_use, flip_ry=False)
            if anchors_info:
                anchor_offsets[:, 0] = -anchor_offsets[:, 0]

        # Convert to anchors
        anchors_to_use = box_3d_encoder.box_3d_to_anchor(
            anchor_boxes_3d_to_use)  # 9910*6
        num_anchors = len(anchors_to_use)  # 9910

        # Project anchors into bev
        bev_anchors, bev_anchors_norm = anchor_projector.project_to_bev(
            anchors_to_use, self._bev_extents)  # 9910*4,9910*4

        # Project box_3d anchors into image space
        img_anchors, img_anchors_norm = \
            anchor_projector.project_to_image_space(
                anchors_to_use, stereo_calib_p2, image_shape)  # 9910*4,9910*4

        # Reorder into [y1, x1, y2, x2] for tf.crop_and_resize op
        self._bev_anchors_norm = bev_anchors_norm[:, [1, 0, 3, 2]]
        self._img_anchors_norm = img_anchors_norm[:, [1, 0, 3, 2]]

        # Fill in placeholder inputs
        self._placeholder_inputs[self.PL_ANCHORS] = anchors_to_use

        # If we are in train/validation mode, and the anchor infos
        # are not empty, store them. Checking for just anchors_ious
        # to be non-empty should be enough.
        if self._train_val_test in ['train', 'val'] and \
                len(anchors_ious) > 0:
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = anchors_ious
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = anchor_offsets
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = anchor_classes

        # During test, or val when there is no anchor info
        elif self._train_val_test in ['test'] or \
                len(anchors_ious) == 0:
            # During testing, or validation with no gt, fill these in with 0s
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = \
                np.zeros(num_anchors)
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = \
                np.zeros([num_anchors, 6])
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = \
                np.zeros(num_anchors)
        else:
            raise ValueError('Got run mode {}, and non-empty anchor info'.
                             format(self._train_val_test))

        self._placeholder_inputs[self.PL_BEV_ANCHORS] = bev_anchors
        self._placeholder_inputs[self.PL_BEV_ANCHORS_NORM] = \
            self._bev_anchors_norm
        self._placeholder_inputs[self.PL_IMG_ANCHORS] = img_anchors
        self._placeholder_inputs[self.PL_IMG_ANCHORS_NORM] = \
            self._img_anchors_norm

    def loss(self, prediction_dict):

        # # these should include mini-batch values only
        # objectness_gt = prediction_dict[self.PRED_MB_OBJECTNESS_GT]
        # offsets_gt = prediction_dict[self.PRED_MB_OFFSETS_GT]
        #
        # # Predictions
        # with tf.variable_scope('rpn_prediction_mini_batch'):
        #     objectness = prediction_dict[self.PRED_MB_OBJECTNESS]
        #     offsets = prediction_dict[self.PRED_MB_OFFSETS]
        #
        # with tf.variable_scope('rpn_losses'):
        #     with tf.variable_scope('objectness'):
        #         cls_loss = losses.WeightedSoftmaxLoss()
        #         cls_loss_weight = self._config.loss_config.cls_loss_weight
        #         objectness_loss = cls_loss(objectness,
        #                                    objectness_gt,
        #                                    weight=cls_loss_weight)
        #
        #         with tf.variable_scope('obj_norm'):
        #             # normalize by the number of anchor mini-batches
        #             objectness_loss = objectness_loss / tf.cast(
        #                 tf.shape(objectness_gt)[0], dtype=tf.float32)
        #             tf.summary.scalar('objectness', objectness_loss)
        #
        #     with tf.variable_scope('regression'):
        #         reg_loss = losses.WeightedSmoothL1Loss()
        #         reg_loss_weight = self._config.loss_config.reg_loss_weight
        #         anchorwise_localization_loss = reg_loss(offsets,
        #                                                 offsets_gt,
        #                                                 weight=reg_loss_weight)
        #         masked_localization_loss = \
        #             anchorwise_localization_loss * objectness_gt[:, 1]
        #         localization_loss = tf.reduce_sum(masked_localization_loss)
        #
        #         with tf.variable_scope('reg_norm'):
        #             # normalize by the number of positive objects
        #             num_positives = tf.reduce_sum(objectness_gt[:, 1])
        #             # Assert the condition `num_positives > 0`
        #             with tf.control_dependencies(
        #                     [tf.assert_positive(num_positives)]):
        #                 localization_loss = localization_loss / num_positives
        #                 tf.summary.scalar('regression', localization_loss)
        #
        #     with tf.variable_scope('total_loss'):
        #         total_loss = objectness_loss + localization_loss
        #
        # loss_dict = {
        #     self.LOSS_RPN_OBJECTNESS: objectness_loss,
        #     self.LOSS_RPN_REGRESSION: localization_loss,
        # }

        # yolo v3 loss
        # label_sbbox = prediction_dict[self.PL_LABEL_SBBOX]
        # label_mbbox = prediction_dict[self.PL_LABEL_MBBOX]
        # label_lbbox = prediction_dict[self.PL_LABEL_LBBOX]
        # true_sbboxes = prediction_dict[self.PL_SBBOXES]
        # true_mbboxes = prediction_dict[self.PL_MBBOXES]
        # true_lbboxes = prediction_dict[self.PL_LBBOXES]
        giou_loss, conf_loss, prob_loss, z_loss = self.model_img.compute_loss(
            self.label_sbbox, self.label_mbbox, self.label_lbbox, self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
        total_loss = giou_loss + conf_loss + prob_loss + z_loss * 1000

        with tf.name_scope('yolov3_2D_detection_loss'):
            # tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss", giou_loss)
            tf.summary.scalar("conf_loss", conf_loss)
            tf.summary.scalar("prob_loss", prob_loss)
            tf.summary.scalar("z_loss", z_loss)
            tf.summary.scalar("yolov3_loss", total_loss)

        loss_dict = {}

        return loss_dict, total_loss

    def create_path_drop_masks(self,
                               p_img,
                               p_bev,
                               random_values):
        """Determines global path drop decision based on given probabilities.

        Args:
            p_img: A tensor of float32, probability of keeping image branch
            p_bev: A tensor of float32, probability of keeping bev branch
            random_values: A tensor of float32 of shape [3], the results
                of coin flips, values should range from 0.0 - 1.0.

        Returns:
            final_img_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
            final_bev_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
        """

        def keep_branch(): return tf.constant(1.0)

        def kill_branch(): return tf.constant(0.0)

        # The logic works as follows:
        # We have flipped 3 coins, first determines the chance of keeping
        # the image branch, second determines keeping bev branch, the third
        # makes the final decision in the case where both branches were killed
        # off, otherwise the initial img and bev chances are kept.

        img_chances = tf.case([(tf.less(random_values[0], p_img),
                                keep_branch)], default=kill_branch)

        bev_chances = tf.case([(tf.less(random_values[1], p_bev),
                                keep_branch)], default=kill_branch)

        # Decision to determine whether both branches were killed off
        third_flip = tf.logical_or(tf.cast(img_chances, dtype=tf.bool),
                                   tf.cast(bev_chances, dtype=tf.bool))
        third_flip = tf.cast(third_flip, dtype=tf.float32)

        # Make a second choice, for the third case
        # Here we use a 50/50 chance to keep either image or bev
        # If its greater than 0.5, keep the image
        img_second_flip = tf.case([(tf.greater(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)
        # If its less than or equal to 0.5, keep bev
        bev_second_flip = tf.case([(tf.less_equal(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)

        # Use lambda since this returns another condition and it needs to
        # be callable
        final_img_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: img_chances)],
                                 default=lambda: img_second_flip)

        final_bev_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: bev_chances)],
                                 default=lambda: bev_second_flip)

        return final_img_mask, final_bev_mask
