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
                      use_bias=use_bias, name=name + '_conv', dilation=dilation)
        bn = tf.contrib.layers.batch_norm(
            inputs=conv, decay=0.9, scale=True, epsilon=1e-5,
            updates_collections=None, is_training=is_training, scope=name + '_batch_norm')
        relu = tf.nn.relu(features=bn, name=name + '_relu')
    return relu


# ResNeXt
def transform_layer(inputs, bottleneck_d, is_training, name, padding='same',
                    use_bias=False, dilation=1, stride=1):
    with tf.variable_scope(name_or_scope=name):
        layer_0 = conv_bn_relu(inputs, bottleneck_d, kernel_size=1, stride=1, is_training=is_training,
                               name=name + '_i', padding=padding, use_bias=use_bias, dilation=dilation)
        layer_1 = conv_bn_relu(layer_0, bottleneck_d, kernel_size=3, stride=stride, is_training=is_training,
                               name=name + '_ii', padding=padding, use_bias=use_bias, dilation=dilation)
        return layer_1


def split_layer(inputs, cardinality, bottleneck_d, is_training, name,
                padding='same', use_bias=False, dilation=1, stride=1):
    with tf.variable_scope(name_or_scope=name):
        layers_group = list()
        for i in range(cardinality):
            layer = transform_layer(inputs, bottleneck_d, is_training, name + '_g' + str(i),
                                    padding, use_bias, dilation, stride=stride)
            layers_group.append(layer)
        concat_dimension = 4  # channels_last
        return tf.concat(layers_group, axis=concat_dimension, name=name)


def transition_layer(inputs, output_channels, is_training, name, padding='same',
                     use_bias=False, dilation=1):
    with tf.variable_scope(name_or_scope=name):
        conv = conv3d(inputs, output_channels, kernel_size=1, stride=1, padding=padding,
                      use_bias=use_bias, name=name + '_iii_conv', dilation=dilation)
        bn = tf.contrib.layers.batch_norm(
            inputs=conv, decay=0.9, scale=True, epsilon=1e-5,
            updates_collections=None, is_training=is_training, scope=name + '_iii_batch_norm')
        return bn


def aggregated_conv(inputs, output_channels, cardinality, bottleneck_d, is_training,
                    name, padding='same', use_bias=False, dilation=1, residual=True, stride=1):
    group = split_layer(inputs, cardinality, bottleneck_d, is_training, name,
                        padding, use_bias, dilation, stride=stride)
    transition = transition_layer(group, output_channels, is_training, name, padding, use_bias, dilation)
    if residual:
        if transition.shape != inputs.shape:
            extension = conv3d(inputs, output_channels, kernel_size=1, stride=stride, padding='same',
                               use_bias=True, name=name + '_residual', dilation=1)
            out = transition + extension
        else:
            out = transition + inputs
    else:
        out = transition
    relu = tf.nn.relu(features=out, name=name + '_iii_relu')
    return relu


def deconv3d(inputs, output_channels, name='deconv'):
    # depth, height and width
    batch, in_depth, in_height, in_width, in_channels = [int(d) for d in inputs.get_shape()]
    dev_filter = tf.get_variable(
        name=name + '/filter', shape=[4, 4, 4, output_channels, in_channels],
        dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        regularizer=slim.l2_regularizer(scale=0.0005))
    deconv = tf.nn.conv3d_transpose(
        value=inputs, filter=dev_filter,
        output_shape=[batch, in_depth * 2, in_height * 2, in_width * 2, output_channels],
        strides=[1, 2, 2, 2, 1], padding='SAME', data_format='NDHWC', name=name)
    '''Further adjustment in strides and filter shape'''
    return deconv


def deconv_bn_relu(inputs, output_channels, is_training, name):
    with tf.variable_scope(name):
        deconv = deconv3d(inputs=inputs, output_channels=output_channels, name=name + '_deconv')
        bn = tf.contrib.layers.batch_norm(inputs=deconv, decay=0.9, scale=True, epsilon=1e-5,
                                          updates_collections=None, is_training=is_training,
                                          scope=name + '_batch_norm')
        relu = tf.nn.relu(features=bn, name=name + '_relu')
    return relu


def residual_block(inputs, output_channels, kernel_size, stride, is_training, name,
                   padding='same', use_bias=False, dilation=1, residual=True):
    with tf.variable_scope(name_or_scope=name):
        # first block
        conv_0 = conv3d(inputs, output_channels, kernel_size, stride, padding=padding,
                        use_bias=use_bias, name=name + '_conv_a', dilation=dilation)
        bn_0 = tf.contrib.layers.batch_norm(inputs=conv_0, decay=0.9, scale=True, epsilon=1e-5,
                                            updates_collections=None, is_training=is_training,
                                            scope=name + '_batch_norm_a')
        relu_0 = tf.nn.relu(features=bn_0, name=name + '_relu_a')
        # second block
        conv_1 = conv3d(relu_0, output_channels, kernel_size, stride=1, padding=padding,
                        use_bias=use_bias, name=name + '_conv_b', dilation=dilation)
        bn_1 = tf.contrib.layers.batch_norm(inputs=conv_1, decay=0.9, scale=True, epsilon=1e-5,
                                            updates_collections=None, is_training=is_training,
                                            scope=name + '_batch_norm_b')
        # shortcut connection
        input_channels = inputs.get_shape().as_list()[-1]
        if input_channels == output_channels:
            if stride == 1:
                shortcut = tf.identity(inputs, name=name + 'shortcut')
            else:
                shortcut = tf.nn.max_pool(input=inputs, ksize=[1, stride, stride, stride, 1],
                                          strides=[1, stride, stride, stride, 1],
                                          padding='VALID', name=name + 'shortcut')
        else:
            shortcut = conv3d(inputs, output_channels, kernel_size=1, stride=stride, padding='same',
                              use_bias=True, name=name + 'shortcut', dilation=1)

        out = bn_1 + shortcut
        relu_1 = tf.nn.relu(features=out, name=name + '_relu_b')
        return relu_1
