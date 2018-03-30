import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv3d(inputs, output_channels, kernel_size, stride, padding='same',
           use_bias=False, name='conv', dilation=1):
    return tf.layers.conv3d(
        inputs=inputs, filters=output_channels, kernel_size=kernel_size,
        strides=stride, padding=padding, data_format='channels_last',
        dilation_rate=(dilation, dilation, dilation),
        activation=None, use_bias=use_bias,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        kernel_regularizer=slim.l2_regularizer(scale=0.0005),
        bias_initializer=tf.zeros_initializer(),
        name=name)


def conv_bn_relu(inputs, output_channels, kernel_size, stride, is_training, name,
                 padding='same', use_bias=False, dilation=1):
    with tf.variable_scope(name_or_scope=name):
        conv = conv3d(inputs, output_channels, kernel_size, stride, padding=padding,
                      use_bias=use_bias, name=name+'_conv', dilation=dilation)
        # bn = tf.contrib.layers.batch_norm(
        #     inputs=conv, decay=0.9, scale=True, epsilon=1e-5,
        #     updates_collections=None, is_training=is_training, scope=name+'_batch_norm')
        relu = tf.nn.relu(features=conv, name=name+'_relu')
    return relu
