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
        self.inputs = None
        self.label = None
        self.domain = None
        self.predicted_label = None
        self.predicted_prob = None
        self.predicted_domain = None
        self.predicted_domain_prob = None
        self.auxiliary1_prob_1x = None
        self.auxiliary2_prob_1x = None
        self.auxiliary3_prob_1x = None

        # frequently used parameters
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

        gpu_number = len(parameter['gpu'].split(','))
        if gpu_number > 1:
            self.device = ['/gpu:0', '/gpu:1', '/cpu:0']
        else:
            self.device = ['/gpu:0', '/gpu:0', '/cpu:0']

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
            # fusion scheme
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
            feature = res_10
            # predicted probability
            predicted_prob = conv3d(inputs=feature, output_channels=self.output_class, kernel_size=1,
                                    stride=1, use_bias=True, name='predicted_prob')
            '''auxiliary prediction'''
            auxiliary3_prob_2x = conv3d(inputs=res_5, output_channels=self.output_class, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary3_prob_2x')
            auxiliary3_prob_1x = deconv3d(inputs=auxiliary3_prob_2x, output_channels=self.output_class,
                                          name='auxiliary3_prob_1x')

            auxiliary2_prob_2x = conv3d(inputs=res_6, output_channels=self.output_class, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary2_prob_2x')
            auxiliary2_prob_1x = deconv3d(inputs=auxiliary2_prob_2x, output_channels=self.output_class,
                                          name='auxiliary2_prob_1x')

            auxiliary1_prob_2x = conv3d(inputs=res_8, output_channels=self.output_class, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary1_prob_2x')
            auxiliary1_prob_1x = deconv3d(inputs=auxiliary1_prob_2x, output_channels=self.output_class,
                                          name='auxiliary1_prob_1x')

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
            domain = tf.contrib.layers.fully_connected(
                inputs=compress_7, num_outputs=2, scope='domain',
                weights_regularizer=tf.contrib.slim.l2_regularizer(scale=0.0005)
            )

        # device: cpu0
        with tf.device(device_name_or_function=self.device[2]):
            softmax_prob = tf.nn.softmax(logits=predicted_prob, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')

            domain_prob = tf.nn.softmax(logits=domain, name='domain_prob')
            predicted_domain = tf.argmax(input=domain_prob, axis=0, name='predicted_domain')

        return predicted_prob, predicted_label, auxiliary1_prob_1x, auxiliary2_prob_1x, auxiliary3_prob_1x, \
            domain, predicted_domain

    def build_model(self):
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, self.input_size, self.input_size,
                                            self.input_size, self.input_channels], name='inputs')
        self.label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.input_size,
                                                           self.input_size, self.input_size],
                                    name='label')
        self.predicted_prob, self.predicted_label, self.auxiliary1_prob_1x, self.auxiliary2_prob_1x, \
            self.auxiliary3_prob_1x, self.domain, self.predicted_domain = self.model(self.inputs)
        '''loss'''
        print('Model built.')

    def train(self):
        pass

    def test(self):
        pass
