import tensorflow as tf

from avod.core.feature_extractors import bev_feature_extractor

slim = tf.contrib.slim


class BevVggPyrDw(bev_feature_extractor.BevFeatureExtractor):
    """Contains modified VGG model definition to extract features from
    Bird's eye view input using pyramid features.
    """

    def _depthwise_separable_conv(self,
                                  inputs,
                                  out_channels,
                                  sc):
        """ Helper function to build the depth-wise separable convolution layer.
        """
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      scope=sc + '/depthwise_conv')
        # bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(depthwise_conv,
                                            out_channels,
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        # bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return pointwise_conv

    def build(self,
              inputs,
              input_pixel_size,
              is_training,
              scope='bev_vgg_pyr'):
        """ Modified VGG for BEV feature extraction with pyramid features
        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.
        Returns:
            The last op containing the log predictions and end_points dict.
        """
        vgg_config = self.config
        dwconv = self._depthwise_separable_conv

        with tf.variable_scope('bev_vgg_pyr', [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d, slim.max_pool2d],
                                outputs_collections=[end_points_collection]):
                with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                                    activation_fn=None,
                                    normalizer_fn=slim.batch_norm):
                    with slim.arg_scope([slim.batch_norm],
                                        is_training=is_training,
                                        activation_fn=tf.nn.relu,
                                        fused=True):

                        # Pad 700 to 704 to allow even divisions for max pooling
                        padded = tf.pad(inputs, [[0, 0], [4, 0], [0, 0], [0, 0]])

                        conv1 = dwconv(padded, vgg_config.vgg_conv1[1], sc='conv1')
                        conv2 = dwconv(conv1, vgg_config.vgg_conv1[1], sc='conv2')
                        pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')

                        conv3 = dwconv(pool1, vgg_config.vgg_conv2[1], sc='conv3')
                        conv4 = dwconv(conv3, vgg_config.vgg_conv2[1], sc='conv4')
                        pool2 = slim.max_pool2d(conv4, [2, 2], scope='pool2')

                        conv5 = dwconv(pool2, vgg_config.vgg_conv3[1], sc='conv5')
                        conv6 = dwconv(conv5, vgg_config.vgg_conv3[1], sc='conv6')
                        conv7 = dwconv(conv6, vgg_config.vgg_conv3[1], sc='conv7')
                        pool3 = slim.max_pool2d(conv7, [2, 2], scope='pool3')

                        conv8 = dwconv(pool3, vgg_config.vgg_conv4[1], sc='conv8')
                        conv9 = dwconv(conv8, vgg_config.vgg_conv4[1], sc='conv9')
                        conv10 = dwconv(conv9, vgg_config.vgg_conv4[1], sc='conv10')

                        # Decoder (upsample and fuse features)
                        upconv3 = slim.conv2d_transpose(
                            conv10,
                            vgg_config.vgg_conv3[1],
                            [3, 3],
                            stride=2,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={
                                'is_training': is_training},
                            scope='upconv3')

                        product3 = tf.concat(
                            (conv7, upconv3), axis=3)
                        pyramid_fusion3 = slim.convolution2d(
                            product3,
                            vgg_config.vgg_conv2[1],
                            [3, 3],
                            scope='pyramid_fusion3')

                        upconv2 = slim.conv2d_transpose(
                            pyramid_fusion3,
                            vgg_config.vgg_conv2[1],
                            [3, 3],
                            stride=2,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={
                                'is_training': is_training},
                            scope='upconv2')

                        product2 = tf.concat(
                            (conv4, upconv2), axis=3)
                        pyramid_fusion_2 = slim.convolution2d(
                            product2,
                            vgg_config.vgg_conv1[1],
                            [3, 3],
                            scope='pyramid_fusion2')

                        upconv1 = slim.conv2d_transpose(
                            pyramid_fusion_2,
                            vgg_config.vgg_conv1[1],
                            [3, 3],
                            stride=2,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={
                                'is_training': is_training},
                            scope='upconv1')

                        product1 = tf.concat(
                            (conv2, upconv1), axis=3)
                        pyramid_fusion1 = slim.convolution2d(
                            product1,
                            vgg_config.vgg_conv1[1],
                            [3, 3],
                            scope='pyramid_fusion1')

                        # Slice off padded area
                        sliced = pyramid_fusion1[:, 4:]

                    feature_maps_out = sliced

                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(
                        end_points_collection)

                    return feature_maps_out, end_points
