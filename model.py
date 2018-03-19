from conv import *
from loss import *
from iostream import *
import os
import numpy as np
import tensorflow as tf
import time


class AdversarialNet:
    def __init__(self, session, parameter):
        self.session = session
        self.parameter_dict = parameter

        # variable declaration
        self.dice_ratio = 0.1
        self.domain_ratio = 0.1
        self.saver = None
        self.inputs = None
        self.label = None
        self.domain_label = None
        self.predicted_feature = None
        self.predicted_label = None
        self.domain_feature = None
        self.predicted_domain = None
        self.auxiliary1_feature_1x = None
        self.auxiliary2_feature_1x = None
        self.auxiliary3_feature_1x = None
        self.main_loss = None
        self.auxiliary1_loss = None
        self.auxiliary2_loss = None
        self.auxiliary3_loss = None
        self.seg_loss = None
        self.adv_loss = None
        self.loss = None

        # frequently used parameters
        gpu_number = len(parameter['gpu'].split(','))
        if gpu_number > 1:
            self.device = ['/gpu:0', '/gpu:1', '/cpu:0']
        else:
            self.device = ['/gpu:0', '/gpu:0', '/cpu:0']
        self.batch_size = parameter['batch_size']
        self.input_size = parameter['input_size']
        self.input_channels = parameter['input_channels']
        self.output_size = parameter['output_size']
        self.output_class = parameter['output_class']
        self.scale = parameter['scale']
        self.feature_size = parameter['feature_size']
        self.cardinality = int(self.feature_size / 4)
        self.sample_from = parameter['sample_from']
        self.sample_to = parameter['sample_to']
        self.phase = parameter['phase']
        self.augmentation = parameter['augmentation']
        self.select_samples = parameter['select_samples']
        # Todo: pay attention to priority of sample selection

        self.build_model()

    def model(self, inputs):
        is_training = (self.phase == 'train')

        with tf.device(device_name_or_function=self.device[0]):
            conv_1 = conv_bn_relu(inputs=inputs, output_channels=self.feature_size, kernel_size=3,
                                  stride=1, is_training=is_training, name='conv_1')
            res_1 = aggregated_conv(inputs=conv_1, output_channels=self.feature_size*2, cardinality=self.cardinality,
                                    bottleneck_d=4, is_training=is_training, name='res_1', padding='same',
                                    use_bias=False, dilation=1)
            pool1 = tf.layers.max_pooling3d(inputs=res_1, pool_size=2, strides=2, name='pool1')
            # pool size?
            res_2 = aggregated_conv(inputs=pool1, output_channels=self.feature_size*4, cardinality=self.cardinality*2,
                                    bottleneck_d=4, is_training=is_training, name='res_2', padding='same',
                                    use_bias=False, dilation=1)
            res_3 = aggregated_conv(inputs=res_2, output_channels=self.feature_size*8, cardinality=self.cardinality*4,
                                    bottleneck_d=4, is_training=is_training, name='res_3', padding='same',
                                    use_bias=False, dilation=2)
            res_4 = aggregated_conv(inputs=res_3, output_channels=self.feature_size*16, cardinality=self.cardinality*8,
                                    bottleneck_d=4, is_training=is_training, name='res_4', padding='same',
                                    use_bias=False, dilation=2)
            res_5 = aggregated_conv(inputs=res_4, output_channels=self.feature_size*16, cardinality=self.cardinality*8,
                                    bottleneck_d=4, is_training=is_training, name='res_5', padding='same',
                                    use_bias=False, dilation=4)
            # Todo: fusion scheme
            fuse_1 = conv_bn_relu(inputs=res_3, output_channels=self.feature_size * 16, kernel_size=1, stride=1,
                                  is_training=is_training, name='fuse_1')
            concat_1 = res_5 + fuse_1
            res_6 = aggregated_conv(inputs=concat_1, output_channels=self.feature_size * 8,
                                    cardinality=self.cardinality * 8, bottleneck_d=4, is_training=is_training,
                                    name='res_6', padding='same', use_bias=False,
                                    dilation=2, residual=True)
            fuse_2 = conv_bn_relu(inputs=res_2, output_channels=self.feature_size * 8, kernel_size=1, stride=1,
                                  is_training=is_training, name='fuse_2')
            concat_2 = res_6 + fuse_2
            res_7 = aggregated_conv(inputs=concat_2, output_channels=self.feature_size * 4,
                                    cardinality=self.cardinality * 4, bottleneck_d=4, is_training=is_training,
                                    name='res_7', padding='same', use_bias=False,
                                    dilation=2, residual=True)
            res_8 = aggregated_conv(inputs=res_7, output_channels=self.feature_size * 4,
                                    cardinality=self.cardinality * 2,
                                    bottleneck_d=4, is_training=is_training,
                                    name='res_8', padding='same', use_bias=False,
                                    dilation=1, residual=False)
            deconv1 = deconv_bn_relu(inputs=res_8, output_channels=self.feature_size * 2, is_training=is_training,
                                     name='deconv1')
            fuse_3 = conv_bn_relu(inputs=res_1, output_channels=self.feature_size * 2, kernel_size=1, stride=1,
                                  is_training=is_training, name='fuse_3')
            concat_3 = deconv1 + fuse_3
            res_9 = aggregated_conv(inputs=concat_3, output_channels=self.feature_size, cardinality=self.cardinality,
                                    bottleneck_d=4, is_training=is_training,
                                    name='res_9', padding='same', use_bias=False,
                                    dilation=1, residual=False)
            res_10 = aggregated_conv(inputs=res_9, output_channels=self.feature_size, cardinality=self.cardinality,
                                     bottleneck_d=4, is_training=is_training,
                                     name='res_10', padding='same', use_bias=False,
                                     dilation=1, residual=False)
            # predicted probability
            predicted_feature = conv3d(inputs=res_10, output_channels=self.output_class, kernel_size=1,
                                    stride=1, use_bias=True, name='predicted_feature')
            '''auxiliary prediction'''
            auxiliary3_feature_2x = conv3d(inputs=res_5, output_channels=self.output_class, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary3_feature_2x')
            auxiliary3_feature_1x = deconv3d(inputs=auxiliary3_feature_2x, output_channels=self.output_class,
                                          name='auxiliary3_feature_1x')

            auxiliary2_feature_2x = conv3d(inputs=res_6, output_channels=self.output_class, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary2_feature_2x')
            auxiliary2_feature_1x = deconv3d(inputs=auxiliary2_feature_2x, output_channels=self.output_class,
                                          name='auxiliary2_feature_1x')

            auxiliary1_feature_2x = conv3d(inputs=res_8, output_channels=self.output_class, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary1_feature_2x')
            auxiliary1_feature_1x = deconv3d(inputs=auxiliary1_feature_2x, output_channels=self.output_class,
                                          name='auxiliary1_feature_1x')

        with tf.device(device_name_or_function=self.device[1]):
            # discriminator
            concat_dimension = 4  # channels_last
            normal_1 = tf.concat([res_1, res_10], axis=concat_dimension, name='normal_1')
            compress_1 = tf.concat([res_3, res_5, res_8], axis=concat_dimension, name='compress_1')

            normal_2 = aggregated_conv(inputs=normal_1, output_channels=self.feature_size*8,
                                       cardinality=self.cardinality*8,
                                       bottleneck_d=4, is_training=is_training,
                                       name='normal_2', padding='same', use_bias=False,
                                       dilation=1, residual=True, stride=2)

            compress_2 = aggregated_conv(inputs=compress_1, output_channels=self.feature_size*8,
                                         cardinality=self.cardinality*8,
                                         bottleneck_d=4, is_training=is_training,
                                         name='compress_2', padding='same', use_bias=False,
                                         dilation=1, residual=True)

            concat_4 = tf.concat([normal_2, compress_2], axis=concat_dimension, name='concat_4')
            compress_3 = aggregated_conv(inputs=concat_4, output_channels=self.feature_size*8,
                                         cardinality=self.cardinality*8,
                                         bottleneck_d=4, is_training=is_training,
                                         name='compress_3', padding='same', use_bias=False,
                                         dilation=1, residual=True)
            compress_4 = aggregated_conv(inputs=compress_3, output_channels=self.feature_size*16,
                                         cardinality=self.cardinality*8,
                                         bottleneck_d=4, is_training=is_training,
                                         name='compress_4', padding='same', use_bias=False,
                                         dilation=1, residual=True, stride=2)
            compress_5 = aggregated_conv(inputs=compress_4, output_channels=self.feature_size*8,
                                         cardinality=self.cardinality*8,
                                         bottleneck_d=4, is_training=is_training,
                                         name='compress_5', padding='same', use_bias=False,
                                         dilation=1, residual=True, stride=2)
            compress_6 = aggregated_conv(inputs=compress_5, output_channels=self.feature_size*4,
                                         cardinality=self.cardinality*4,
                                         bottleneck_d=4, is_training=is_training,
                                         name='compress_6', padding='same', use_bias=False,
                                         dilation=1, residual=True, stride=2)
            compress_7 = conv_bn_relu(inputs=compress_6, output_channels=self.feature_size*4, kernel_size=1, stride=1,
                                      is_training=is_training, name='compress_7', use_bias=True)
            # Todo: average pooling?
            average = tf.reduce_mean(input_tensor=compress_7, axis=[1, 2, 3], name='average_pooling')
            domain_feature = tf.contrib.layers.fully_connected(
                inputs=average, num_outputs=2, scope='domain',
                weights_regularizer=tf.contrib.slim.l2_regularizer(scale=0.0005))

        # device: cpu0
        with tf.device(device_name_or_function=self.device[2]):
            softmax_prob = tf.nn.softmax(logits=predicted_feature, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')

            domain_prob = tf.nn.softmax(logits=domain_feature, name='domain_prob')
            predicted_domain = tf.argmax(input=domain_prob, axis=0, name='predicted_domain')

        return predicted_feature, predicted_label, auxiliary1_feature_1x, auxiliary2_feature_1x, \
            auxiliary3_feature_1x, domain_feature, predicted_domain

    def build_model(self):
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, self.input_size, self.input_size,
                                            self.input_size, self.input_channels], name='inputs')
        self.label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.input_size,
                                                           self.input_size, self.input_size],
                                    name='label')
        self.domain_label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='domain_label')

        self.predicted_feature, self.predicted_label, self.auxiliary1_feature_1x, self.auxiliary2_feature_1x, \
            self.auxiliary3_feature_1x, self.domain_feature, self.predicted_domain = self.model(self.inputs)

        self.main_loss = cross_entropy_loss(self.predicted_feature, self.label, self.output_class) + \
            self.dice_ratio * dice_loss(self.predicted_feature, self.label, self.output_class)
        self.auxiliary1_loss = cross_entropy_loss(self.auxiliary1_feature_1x, self.label, self.output_class) + \
            self.dice_ratio * dice_loss(self.auxiliary1_feature_1x, self.label, self.output_class)
        self.auxiliary2_loss = cross_entropy_loss(self.auxiliary2_feature_1x, self.label, self.output_class) + \
            self.dice_ratio * dice_loss(self.auxiliary2_feature_1x, self.label, self.output_class)
        self.auxiliary3_loss = cross_entropy_loss(self.auxiliary3_feature_1x, self.label, self.output_class) + \
            self.dice_ratio * dice_loss(self.auxiliary3_feature_1x, self.label, self.output_class)

        self.seg_loss = (self.main_loss + 0.8 * self.auxiliary1_loss + 0.4 * self.auxiliary2_loss +
                         0.2 * self.auxiliary3_loss) / 2.4
        self.adv_loss = domain_loss(self.domain_feature, self.domain_label, 2)
        self.loss = self.seg_loss - self.domain_ratio * self.adv_loss

        self.saver = tf.train.Saver(max_to_keep=20)
        print('Model built.')

    def train(self):
        # sample selection
        # epoch
        pass

    def test(self):
        # write function for testing
        # split
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass
