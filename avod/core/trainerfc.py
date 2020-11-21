"""Detection model trainer.

This file provides a generic training method to train a
DetectionModel.
"""
import datetime
import os
import tensorflow as tf
import time
import numpy as np
import shutil

from avod.builders import optimizer_builder
from avod.core import trainer_utils
from avod.core import summary_utils

slim = tf.contrib.slim


def train(model, train_config):
    """Training function for detection models.

    Args:
        model: The detection model object.
        train_config: a train_*pb2 protobuf.
            training i.e. loading RPN weights onto AVOD model.
    """

    model = model
    train_config = train_config
    # Get model configurations
    model_config = model.model_config

    # Create a variable tensor to hold the global step
    # global_step_tensor = tf.Variable(
    #     0, trainable=False, name='global_step')

    #############################
    # Get training configurations
    #############################
    max_iterations = train_config.max_iterations
    summary_interval = train_config.summary_interval
    checkpoint_interval = \
        train_config.checkpoint_interval
    max_checkpoints = train_config.max_checkpoints_to_keep

    paths_config = model_config.paths_config
    logdir = paths_config.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    checkpoint_dir = paths_config.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = checkpoint_dir + '/' + \
        model_config.checkpoint_name

    global_summaries = set([])

    # The model should return a dictionary of predictions
    prediction_dict = model.build()

    summary_histograms = train_config.summary_histograms
    summary_img_images = train_config.summary_img_images
    summary_bev_images = train_config.summary_bev_images

    ##############################
    # Setup loss
    ##############################
    losses_dict, total_loss = model.loss(prediction_dict)

    # Optimizer
    # training_optimizer = optimizer_builder.build(
    #     train_config.optimizer,
    #     global_summaries,
    #     global_step_tensor)
    #
    # # Create the train op
    # # with tf.variable_scope('train_op'):
    # train_op = slim.learning.create_train_op(
    #     total_loss,
    #     training_optimizer,
    #     clip_gradient_norm=1.0,
    #     global_step=global_step_tensor,
    #     update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))  #

    global_step_tensor = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
    with tf.name_scope('learn_rate'):
        warmup_steps = tf.constant(2 * 6733,
                                   dtype=tf.float64, name='warmup_steps')
        train_steps = tf.constant((20 + 30) * 6733,
                                  dtype=tf.float64, name='train_steps')
        learn_rate = tf.cond(
            pred=global_step_tensor < warmup_steps,
            true_fn=lambda: global_step_tensor / warmup_steps * 1e-4,
            false_fn=lambda: 1e-6 + 0.5 * (1e-4 - 1e-6) * (1 + tf.cos(
                (global_step_tensor - warmup_steps) / (train_steps - warmup_steps) * np.pi)))
        tf.summary.scalar("learn_rate", learn_rate)
        global_step_update = tf.assign_add(global_step_tensor, 1.0)

    with tf.name_scope("define_weight_decay"):
        moving_ave = tf.train.ExponentialMovingAverage(0.9995).apply(tf.trainable_variables())

    with tf.name_scope("define_first_stage_train"):
        var = tf.trainable_variables()
        trainable_all_name = ['box_predictor']
        first_stage_trainable_var_list = [val for val in var if str(val.name).split('/')[0] in trainable_all_name]
        first_stage_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(
            total_loss, var_list=first_stage_trainable_var_list)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                with tf.control_dependencies([moving_ave]):
                    train_op_with_frozen_variables = tf.no_op()

    with tf.name_scope("define_second_stage_train"):
        second_stage_trainable_var_list = tf.trainable_variables()
        second_stage_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(
            total_loss, var_list=second_stage_trainable_var_list)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                with tf.control_dependencies([moving_ave]):
                    train_op_with_all_variables = tf.no_op()
    train_op = train_op_with_frozen_variables

    # Save checkpoints regularly.
    saver = tf.train.Saver(max_to_keep=max_checkpoints,
                           pad_step_number=True)

    # Add the result of the train_op to the summary
    tf.summary.scalar("training_loss", total_loss)

    # Add maximum memory usage summary op
    # This op can only be run on device with gpu
    # so it's skipped on travis
    is_travis = 'TRAVIS' in os.environ
    if not is_travis:
        # tf.summary.scalar('bytes_in_use',
        #                   tf.contrib.memory_stats.BytesInUse())
        tf.summary.scalar('max_bytes',
                          tf.contrib.memory_stats.MaxBytesInUse())

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_merged = summary_utils.summaries_to_keep(
        summaries,
        global_summaries,
        histograms=summary_histograms,
        input_imgs=summary_img_images,
        input_bevs=summary_bev_images
    )

    allow_gpu_mem_growth = train_config.allow_gpu_mem_growth
    if allow_gpu_mem_growth:
        # GPU memory config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = allow_gpu_mem_growth
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()

    # Create unique folder name using datetime for summary writer
    datetime_str = str(datetime.datetime.now())
    logdir = logdir + '/train'
    train_writer = tf.summary.FileWriter(logdir + '/' + datetime_str,
                                         sess.graph)

    # Create init op
    init = tf.global_variables_initializer()

    # Continue from last saved checkpoint
    if not train_config.overwrite_checkpoints:
        trainer_utils.load_checkpoints(checkpoint_dir,
                                       saver)
        if len(saver.last_checkpoints) > 0:
            checkpoint_to_restore = saver.last_checkpoints[-1]
            saver.restore(sess, checkpoint_to_restore)
            print('avod checkpoints')
        else:
            # Initialize the variables
            # sess.run(init)
            var = tf.global_variables()
            all_name = ['box_predictor']
            # all_name.extend(conv_layer)
            var_to_restore = [val for val in var if str(val.name).split('/')[0] not in all_name]
            print('var_to_restore:', len(var_to_restore))
            saver_in = tf.train.Saver(var_to_restore)
            saver_in.restore(
                sess, '/media/personal_data/zhangye/previous_results/avod_results062507020304/pyramid_cars_with_aug_example-00269320')
            # qqq = sess.run(tf.report_uninitialized_variables())
            # print('utill not init:', len(qqq))
            var_to_init = [val for val in var if str(val.name).split('/')[0] in all_name]
            print('var_to_init:', len(var_to_init))
            init_new_vars_op = tf.initialize_variables(var_to_init)
            sess.run(init_new_vars_op)
            print('yolov3 checkpoints(image)')
    else:
        # Initialize the variables
        sess.run(init)

    # Read the global step if restored
    global_step = tf.train.global_step(sess,
                                       global_step_tensor)
    print('Starting from step {} / {}'.format(
        global_step, max_iterations))
    # saver.save(sess,
    #            save_path=checkpoint_path,
    #            global_step=global_step)
    # print('Step {} / {}, Checkpoint saved to {}-{:08d}'.format(
    #     global_step, max_iterations,
    #     checkpoint_path, global_step))

    # Main Training Loop
    last_time = time.time()

    predicted_dir_path = '/home/zhangy/yolov3/mAP/predicted'
    ground_truth_dir_path = '/home/zhangy/yolov3/mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)

    for step in range(global_step, max_iterations + 1):

        # Save checkpoint
        if step % checkpoint_interval == 0:
            global_step = tf.train.global_step(sess,
                                               global_step_tensor)
            # if global_step / checkpoint_interval > 39:
            #     train_op = train_op_with_all_variables

            saver.save(sess,
                       save_path=checkpoint_path,
                       global_step=global_step)

            print('Step {} / {}, Checkpoint saved to {}-{:08d}'.format(
                step, max_iterations,
                checkpoint_path, global_step))

        # Create feed_dict for inferencing
        feed_dict = model.create_feed_dict()

        # Write summaries and train op
        if step % summary_interval == 0:
            current_time = time.time()
            time_elapsed = current_time - last_time
            last_time = current_time

            train_op_loss, _, summary_out = sess.run(
                [total_loss, train_op, summary_merged], feed_dict=feed_dict)

            print('Step {}, Total Loss {:0.3f}, Time Elapsed {:0.3f} s'.format(
                step, train_op_loss, time_elapsed))
            train_writer.add_summary(summary_out, step)

        else:
            # Run the train op only
            sess.run([total_loss, train_op], feed_dict)

    # Close the summary writers
    train_writer.close()
