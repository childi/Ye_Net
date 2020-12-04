#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 17:50:26
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3_z import YOLOV3
from core.config import cfg
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./data/log/train"
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess                = tf.Session(config=config)
#        self.sess                = tf.Session(allow_soft_placement=True)

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 4], name='input_data')
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable    = tf.placeholder(dtype=tf.bool, name='training')
            self.p2_I = tf.placeholder(dtype=tf.float32, shape=[None, 4, 3], name='input_data')

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable,
                                self.label_sbbox, self.label_mbbox, self.label_lbbox)  # self.p2_I,
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss, self.z_loss = self.model.compute_loss(
                self.label_sbbox, self.label_mbbox, self.label_lbbox,
                self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)  # , self.alpha_loss
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss + self.z_loss  #  * 1000 + self.alpha_loss * 10

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            # self.learn_rate = self.learn_rate_end + 0.5 * (8.86*1e-5 - self.learn_rate_end) * (1 + tf.cos(
            #     (self.global_step / train_steps) * np.pi))  # 7.85*1e-6
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=40)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate",      self.learn_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("z_loss", self.z_loss)
            # tf.summary.scalar("alpha_loss", self.alpha_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "./data/log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer_train = tf.summary.FileWriter(logdir+'train', graph=self.sess.graph)
            self.summary_writer_test = tf.summary.FileWriter(logdir+'test')

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
            self.first_stage_epochs = 0  #
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        logdir_path = '/home/zhangy/yolov3/data/logdir.txt'
        if os.path.exists(logdir_path): os.remove(logdir_path)

        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []
            giou_epoch_loss, conf_epoch_loss, prob_epoch_loss, z_epoch_loss, alpha_epoch_loss = [], [], [], [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbboxes: train_data[4],
                                                self.true_mbboxes: train_data[5],
                                                self.true_lbboxes: train_data[6],
                                                self.trainable:    True,
                                                self.p2_I:         train_data[7]
                })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer_train.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" % train_step_loss)

            for test_data in self.testset:
                test_step_loss, summary_test, giou_loss, conf_loss, prob_loss, z_loss = self.sess.run(
                    [self.loss, self.write_op, self.giou_loss, self.conf_loss, self.prob_loss, self.z_loss],
                    feed_dict={
                        self.input_data:   test_data[0],
                        self.label_sbbox:  test_data[1],
                        self.label_mbbox:  test_data[2],
                        self.label_lbbox:  test_data[3],
                        self.true_sbboxes: test_data[4],
                        self.true_mbboxes: test_data[5],
                        self.true_lbboxes: test_data[6],
                        self.trainable:    False,
                        self.p2_I:         test_data[7]
                })  # , alpha_loss  # , self.alpha_loss

                test_epoch_loss.append(test_step_loss)
                giou_epoch_loss.append(giou_loss)
                conf_epoch_loss.append(conf_loss)
                prob_epoch_loss.append(prob_loss)
                z_epoch_loss.append(z_loss)
                # alpha_epoch_loss.append(alpha_loss)
                self.summary_writer_test.add_summary(summary_test, global_step_val)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "/media/personal_data/zhangye/yolov3_ckpt/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            #  alpha_epoch_loss: %.8f  # , np.mean(alpha_epoch_loss)
            annotation = "=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..." \
                         % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file)
            annotation1 = 'giou_epoch_loss: %.2f conf_epoch_loss: %.2f prob_epoch_loss: %.2f z_epoch_loss: %.8f'\
                          % (np.mean(giou_epoch_loss), np.mean(conf_epoch_loss), np.mean(prob_epoch_loss), np.mean(z_epoch_loss))
            print(annotation)
            print(annotation1)
            with open(logdir_path, 'a') as a:
                a.write(annotation + '\n')
                a.write(annotation1 + '\n')
            self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == '__main__': YoloTrain().train()
