from conv import *
from lossfun import *
from iostream import *
import os
import numpy as np
import tensorflow as tf
import time


class AdversarialNet:
    def __init__(self, session, parameter):
        self.session = session
        self.parameter = parameter

        # variable declaration
        self.dice_ratio = 0.1
        self.domain_ratio = None
        self.saver = None
        self.trainable_variables = None
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
        self.mix_loss = None
        self.slice = None

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
        self.epoch = self.parameter['epoch']
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
        self.domain_ratio = tf.placeholder(dtype=tf.float32, shape=[1], name='domain_ratio')

        self.predicted_feature, self.predicted_label, self.auxiliary1_feature_1x, self.auxiliary2_feature_1x, \
            self.auxiliary3_feature_1x, self.domain_feature, self.predicted_domain = self.model(self.inputs)

        self.slice = [i for i in range(self.domain_label.shape[0])
                      if self.domain_label[i] == tf.constant(0, dtype=tf.int32)]
        self.predicted_feature = self.predicted_feature[self.slice]

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
        self.mix_loss = self.seg_loss - self.domain_ratio * self.adv_loss

        self.trainable_variables = tf.trainable_variables()
        self.saver = tf.train.Saver(max_to_keep=20)
        print('Model built.')

    def train(self):
        learning_rate = self.parameter['learning_rate']
        beta1 = self.parameter['beta1']

        # dynamics problem -> placeholder
        seg_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(
            self.seg_loss, var_list=self.trainable_variables)
        adv_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(
            self.adv_loss, var_list=self.trainable_variables)
        mix_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(
            self.mix_loss, var_list=self.trainable_variables)

        self.session.run(tf.global_variables_initializer())

        # Todo: load pre-trained model and checkpoint

        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        # log_writer = tf.summary.FileWriter(logdir='logs/', graph=self.session.graph)

        source_image_filelist, source_label_filelist = generate_filelist(
            self.parameter['source_data_dir'], self.parameter['source_label_dir'])
        source_domain_info = [0] * len(source_label_filelist)
        # sample selection
        target_image_filelist, target_label_filelist = generate_filelist(
            self.parameter['target_data_dir'], self.parameter['target_label_dir'])
        target_image_filelist, target_label_filelist = self.sample_selection(
            target_image_filelist, target_label_filelist)
        target_domain_info = [1] * len(target_label_filelist)

        mix_image_filelist = source_image_filelist + target_image_filelist
        mix_label_filelist = source_label_filelist + target_label_filelist
        mix_domain_info = source_domain_info + target_domain_info

        if not os.path.exists('loss/'):
            os.makedirs('loss/')
        line_buffer = 1
        with open(file='loss/loss_{}.txt'.format(self.parameter['name']),
                  mode='w', buffering=line_buffer) as loss_log:
            loss_log.write('[Train Mode]\n')
            loss_log.write(write_json(self.parameter))
            loss_log.write('\n')

            for epoch in range(self.epoch):
                domain_ratio = np.array([0.1], dtype=np.float32)
                condition1 = True
                condition2 = True
                # Todo: template
                if condition1:
                    self.train_task(mix_image_filelist, mix_label_filelist, mix_domain_info,
                                    mix_optimizer, domain_ratio, loss_log, epoch)
                elif condition2:
                    self.train_task(source_image_filelist, source_label_filelist, source_domain_info,
                                    seg_optimizer, domain_ratio, loss_log, epoch)
                else:
                    self.train_task(mix_image_filelist, mix_image_filelist, mix_image_filelist,
                                    adv_optimizer, domain_ratio, loss_log, epoch)
                # save and test module
                if np.mod(epoch + 1, self.parameter['save_interval']) == 0:
                    self.save_checkpoint(self.parameter['checkpoint_dir'],
                                         self.parameter['model_name'], global_step=epoch + 1)
                    print('[Save] Model saved with epoch %d' % (epoch + 1))
                if np.mod(epoch + 1, self.parameter['test_interval']) == 0:
                    pass
        pass

    def train_task(self, train_image_filelist, train_label_filelist, train_domain_info,
                   optimizer, domain_ratio, loss_log, epoch):
        start_time = time.time()
        image_batch, label_batch, domain_batch = load_train_batches(
            train_image_filelist, train_label_filelist, train_domain_info, self.input_size,
            self.batch_size, flipping=True, rotation=True, scale=self.scale)
        # update network

        _, seg_loss, adv_loss, mix_loss = self.session.run(
            [optimizer, self.seg_loss, self.adv_loss, self.mix_loss],
            feed_dict={self.inputs: image_batch, self.label: label_batch,
                       self.domain_label: domain_batch, self.domain_ratio: domain_ratio})

        '''temp'''
        string_format = f'[label] {str(np.unique(label_batch))} \n'
        string_format += '[Epoch] %d, time: %4.4f [Loss] loss: %.8f \n [Loss] seg_loss: %.8f, adv_loss: %.8f \n\n' \
                         % (epoch + 1, time.time() - start_time, mix_loss, seg_loss, adv_loss)
        loss_log.write(string_format)
        print(string_format, end='')

    def test(self):
        # write function for testing
        # split
        pass

    def sample_selection(self, image_filelist, label_filelist):
        selected_image_filelist = []
        selected_label_filelist = []
        if self.select_samples is not None:
            for var in self.select_samples:
                selected_image_filelist.append(image_filelist[var])
                selected_label_filelist.append(label_filelist[var])
        else:
            selected_image_filelist = image_filelist[self.sample_from: self.sample_to + 1]
            selected_label_filelist = label_filelist[self.sample_from: self.sample_to + 1]
        return selected_image_filelist, selected_label_filelist

    def save_checkpoint(self, checkpoint_dir, model_name, global_step):
        model_dir = 'model_{}_{}_{}'.format(self.feature_size, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # function of global_step?
        self.saver.save(self.session, os.path.join(checkpoint_dir, model_name), global_step=global_step)

    def load_checkpoint(self, checkpoint_dir):
        model_dir = 'model_{}_{}_{}'.format(self.feature_size, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoint_name))
            return True
        else:
            return False
