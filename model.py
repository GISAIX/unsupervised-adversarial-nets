from conv import *
from inference import Evaluation, infer
from iostream import *
from lossfun import *
import os
import numpy as np
import tensorflow as tf
import time


class AdversarialNet:
    def __init__(self, session, parameter):
        self.session = session
        self.parameter = parameter

        # variable declaration
        self.inputs = None
        self.label = None
        self.domain = None
        self.coefficient = None
        self.predicted_feature = None
        self.predicted_label = None
        self.auxiliary1_feature_1x = None
        self.auxiliary2_feature_1x = None
        self.auxiliary3_feature_1x = None
        self.domain_prob = None
        self.domain_feature = None
        self.predicted_domain = None
        # loss
        self.main_entropy = None
        self.auxiliary1_entropy = None
        self.auxiliary2_entropy = None
        self.auxiliary3_entropy = None
        self.seg_entropy = None
        self.main_dice = None
        self.auxiliary1_dice = None
        self.auxiliary2_dice = None
        self.auxiliary3_dice = None
        self.seg_dice = None
        self.dis_loss = None
        self.seg_loss = None
        # variable
        self.trainable_variables = None
        self.seg_variables = None
        self.dis_variables = None
        self.saver = None

        # frequently used parameters
        gpu_number = len(parameter['gpu'].split(','))
        if gpu_number > 1:
            self.device = ['/gpu:0', '/gpu:1', '/cpu:0']
        else:
            self.device = ['/gpu:0', '/gpu:0', '/cpu:0']
        self.batch_size = parameter['batch_size']
        self.input_size = parameter['input_size']
        self.test_stride = parameter['test_stride']
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
        self.iteration = self.parameter['iteration']
        # Todo: pay attention to priority of sample selection

        self.build_model()

    def model(self, inputs, runtime_batch_size):
        concat_dimension = 4  # channels_last
        is_training = (self.phase == 'train')
        with tf.device(device_name_or_function=self.device[0]):
            with tf.variable_scope('seg'):
                conv_1 = conv_bn_relu(inputs=inputs, output_channels=self.feature_size, kernel_size=7, stride=1,
                                      is_training=is_training, name='conv_1')
                res_1 = aggregated_conv(inputs=conv_1, output_channels=self.feature_size * 2,
                                        cardinality=self.cardinality, bottleneck_d=4, is_training=is_training,
                                        name='res_1', padding='same', use_bias=False, dilation=1)
                pool1 = tf.layers.max_pooling3d(inputs=res_1, pool_size=2, strides=2, name='pool_0')

                res_2 = aggregated_conv(inputs=pool1, output_channels=self.feature_size * 4,
                                        cardinality=self.cardinality * 2, bottleneck_d=4, is_training=is_training,
                                        name='res_2', padding='same', use_bias=False, dilation=1)
                res_3 = aggregated_conv(inputs=res_2, output_channels=self.feature_size * 8,
                                        cardinality=self.cardinality * 4, bottleneck_d=4, is_training=is_training,
                                        name='res_3', padding='same', use_bias=False, dilation=2)
                res_4 = aggregated_conv(inputs=res_3, output_channels=self.feature_size * 16,
                                        cardinality=self.cardinality * 8, bottleneck_d=4, is_training=is_training,
                                        name='res_4', padding='same', use_bias=False, dilation=4)
                # Todo: fusion scheme
                # fuse_1 = conv_bn_relu(inputs=res_3, output_channels=self.feature_size * 16, kernel_size=1, stride=1,
                #                       is_training=is_training, name='fuse_1')
                # concat_1 = res_4 + fuse_1
                concat_1 = tf.concat([res_4, res_3], axis=concat_dimension, name='concat_1')
                res_5 = aggregated_conv(inputs=concat_1, output_channels=self.feature_size * 8,
                                        cardinality=self.cardinality * 4, bottleneck_d=4, is_training=is_training,
                                        name='res_5', padding='same', use_bias=False, dilation=2, residual=True)
                # fuse_2 = conv_bn_relu(inputs=res_2, output_channels=self.feature_size * 8, kernel_size=1, stride=1,
                #                       is_training=is_training, name='fuse_2')
                # concat_2 = res_6 + fuse_2
                concat_2 = tf.concat([res_5, res_2], axis=concat_dimension, name='concat_2')
                res_6 = aggregated_conv(inputs=concat_2, output_channels=self.feature_size * 4,
                                        cardinality=self.cardinality * 2, bottleneck_d=4, is_training=is_training,
                                        name='res_6', padding='same', use_bias=False, dilation=1, residual=True)
                deconv1 = deconv_bn_relu(inputs=res_6, output_channels=self.feature_size * 4, is_training=is_training,
                                         name='deconv1', runtime_batch_size=runtime_batch_size)
                # fuse_3 = conv_bn_relu(inputs=res_1, output_channels=self.feature_size * 2, kernel_size=1, stride=1,
                #                       is_training=is_training, name='fuse_3')
                # concat_3 = deconv1 + fuse_3
                concat_3 = tf.concat([deconv1, res_1], axis=concat_dimension, name='concat_3')
                res_7 = aggregated_conv(inputs=concat_3, output_channels=self.feature_size,
                                        cardinality=self.cardinality, bottleneck_d=4, is_training=is_training,
                                        name='res_9', padding='same', use_bias=False, dilation=1, residual=False)
                feature = res_7
                # predicted probability
                predicted_feature = conv3d(inputs=feature, output_channels=self.output_class, kernel_size=1, stride=1,
                                           use_bias=True, name='predicted_feature')
                '''auxiliary prediction'''

                auxiliary3_feature_2x = deconv3d(inputs=res_4, output_channels=self.feature_size,
                                                 name='auxiliary3_feature_2x', runtime_batch_size=runtime_batch_size)
                auxiliary3_feature_1x = conv3d(inputs=auxiliary3_feature_2x, output_channels=self.output_class,
                                               kernel_size=1, stride=1, use_bias=True, name='auxiliary3_feature_1x')

                auxiliary2_feature_2x = deconv3d(inputs=res_5, output_channels=self.feature_size,
                                                 name='auxiliary2_feature_2x', runtime_batch_size=runtime_batch_size)
                auxiliary2_feature_1x = conv3d(inputs=auxiliary2_feature_2x, output_channels=self.output_class,
                                               kernel_size=1, stride=1, use_bias=True, name='auxiliary2_feature_1x')

                auxiliary1_feature_2x = deconv3d(inputs=res_6, output_channels=self.feature_size,
                                                 name='auxiliary1_feature_2x', runtime_batch_size=runtime_batch_size)
                auxiliary1_feature_1x = conv3d(inputs=auxiliary1_feature_2x, output_channels=self.output_class,
                                               kernel_size=1, stride=1, use_bias=True, name='auxiliary1_feature_1x')

        with tf.device(device_name_or_function=self.device[1]):
            with tf.variable_scope('dis'):
                extracted_feature = tf.concat([feature, auxiliary3_feature_2x, auxiliary2_feature_2x,
                                               auxiliary1_feature_2x], axis=concat_dimension, name='extracted_feature')
                # extracted_feature = inputs
                filters = [64, 64, 128, 256, 512]
                strides = [1, 1, 2, 1, 2]

                dis1_1 = conv_bn_relu(inputs=extracted_feature, output_channels=filters[0], kernel_size=7,
                                      stride=strides[0], is_training=is_training, name='dis1_1')
                pool1_2 = tf.layers.max_pooling3d(inputs=dis1_1, pool_size=3, strides=2, name='pool1_2', padding='same')

                dis2_1 = residual_block(inputs=pool1_2, output_channels=filters[1], kernel_size=3,
                                        stride=strides[1], is_training=is_training, name='dis2_1',
                                        padding='same', use_bias=False, dilation=1, residual=True)
                dis2_2 = residual_block(inputs=dis2_1, output_channels=filters[1], kernel_size=3,
                                        stride=1, is_training=is_training, name='dis2_2',
                                        padding='same', use_bias=False, dilation=1, residual=True)

                dis3_1 = residual_block(inputs=dis2_2, output_channels=filters[2], kernel_size=3,
                                        stride=strides[2], is_training=is_training, name='dis3_1',
                                        padding='same', use_bias=False, dilation=1, residual=True)
                dis3_2 = residual_block(inputs=dis3_1, output_channels=filters[2], kernel_size=3,
                                        stride=1, is_training=is_training, name='dis3_2',
                                        padding='same', use_bias=False, dilation=1, residual=True)

                dis4_1 = residual_block(inputs=dis3_2, output_channels=filters[3], kernel_size=3,
                                        stride=strides[3], is_training=is_training, name='dis4_1',
                                        padding='same', use_bias=False, dilation=1, residual=True)
                dis4_2 = residual_block(inputs=dis4_1, output_channels=filters[3], kernel_size=3,
                                        stride=1, is_training=is_training, name='dis4_2',
                                        padding='same', use_bias=False, dilation=1, residual=True)

                dis5_1 = residual_block(inputs=dis4_2, output_channels=filters[4], kernel_size=3,
                                        stride=strides[4], is_training=is_training, name='dis5_1',
                                        padding='same', use_bias=False, dilation=1, residual=True)
                dis5_2 = residual_block(inputs=dis5_1, output_channels=filters[4], kernel_size=3,
                                        stride=1, is_training=is_training, name='dis5_2',
                                        padding='same', use_bias=False, dilation=1, residual=True)

                global_average = tf.reduce_mean(dis5_2, [1, 2, 3])
                fc_1 = tf.contrib.layers.fully_connected(
                    inputs=global_average, num_outputs=64, scope='fc_1', activation_fn=tf.nn.relu,
                    weights_regularizer=tf.contrib.slim.l2_regularizer(scale=0.0005))
                domain_feature = tf.contrib.layers.fully_connected(
                    inputs=fc_1, num_outputs=2, scope='domain_feature', activation_fn=None,
                    weights_regularizer=tf.contrib.slim.l2_regularizer(scale=0.0005))

        # device: cpu0
        with tf.device(device_name_or_function=self.device[2]):
            softmax_prob = tf.nn.softmax(logits=predicted_feature, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')

            domain_prob = tf.nn.softmax(logits=domain_feature, name='domain_prob')
            predicted_domain = tf.argmax(input=domain_prob, axis=1, name='predicted_domain')

        return predicted_feature, predicted_label, auxiliary1_feature_1x, auxiliary2_feature_1x, \
            auxiliary3_feature_1x, domain_feature, domain_prob, predicted_domain

    def build_model(self):
        # use None to replace batch size
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size,
                                                              self.input_size, self.input_channels], name='inputs')
        self.label = tf.placeholder(dtype=tf.int32, shape=[None, self.input_size, self.input_size,
                                                           self.input_size], name='label')
        self.domain = tf.placeholder(dtype=tf.int32, shape=[None], name='domain')
        self.coefficient = tf.placeholder(dtype=tf.float32, shape=[2], name='coefficient')
        # 0: dice coefficient, 1: discriminative ratio

        runtime_batch_size = tf.shape(self.inputs)[0]

        self.predicted_feature, self.predicted_label, self.auxiliary1_feature_1x, self.auxiliary2_feature_1x, \
            self.auxiliary3_feature_1x, self.domain_feature, self.domain_prob, self.predicted_domain = self.model(
                self.inputs, runtime_batch_size)

        self.main_entropy = cross_entropy_loss(self.predicted_feature, self.label, self.output_class)
        self.auxiliary1_entropy = cross_entropy_loss(self.auxiliary1_feature_1x, self.label, self.output_class)
        self.auxiliary2_entropy = cross_entropy_loss(self.auxiliary2_feature_1x, self.label, self.output_class)
        self.auxiliary3_entropy = cross_entropy_loss(self.auxiliary3_feature_1x, self.label, self.output_class)
        self.seg_entropy = (self.main_entropy + 0.9 * self.auxiliary1_entropy + 0.6 * self.auxiliary2_entropy +
                            0.3 * self.auxiliary3_entropy) / 2.8

        self.main_dice = dice_loss(self.predicted_feature, self.label, self.output_class)
        self.auxiliary1_dice = dice_loss(self.auxiliary1_feature_1x, self.label, self.output_class)
        self.auxiliary2_dice = dice_loss(self.auxiliary2_feature_1x, self.label, self.output_class)
        self.auxiliary3_dice = dice_loss(self.auxiliary3_feature_1x, self.label, self.output_class)
        self.seg_dice = (self.main_dice + 0.8 * self.auxiliary1_dice + 0.4 * self.auxiliary2_dice +
                         0.2 * self.auxiliary3_dice) / 2.4

        self.dis_loss = discriminative_loss(self.domain_feature, self.domain)
        self.seg_loss = self.seg_entropy + self.seg_dice * self.coefficient[0] - self.dis_loss * self.coefficient[1]

        self.trainable_variables = tf.trainable_variables()
        self.seg_variables = tf.trainable_variables(scope='seg')
        self.dis_variables = tf.trainable_variables(scope='dis')
        self.saver = tf.train.Saver(max_to_keep=20)
        print('Model built.')

    def train(self):
        learning_rate_dis = self.parameter['learning_rate_dis']
        beta1_dis = self.parameter['beta1_dis']
        learning_rate_seg = self.parameter['learning_rate_seg']
        beta1_seg = self.parameter['beta1_seg']

        dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_dis, beta1=beta1_dis).minimize(
            self.dis_loss, var_list=self.dis_variables)
        seg_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_seg, beta1=beta1_seg).minimize(
            self.seg_loss, var_list=self.seg_variables)
        self.session.run(tf.global_variables_initializer())

        # Todo: load pre-trained model and checkpoint

        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        # log_writer = tf.summary.FileWriter(logdir='logs/', graph=self.session.graph)

        # 0: source, 1: target
        source_image_filelist, source_label_filelist = generate_filelist(
            self.parameter['source_data_dir'], self.parameter['source_label_dir'])
        source_domain_list = [0] * len(source_label_filelist)

        target_image_filelist, target_label_filelist = generate_filelist(
            self.parameter['target_data_dir'], self.parameter['target_label_dir'])
        target_domain_list = [1] * len(target_label_filelist)

        # load all images to save time
        start_time = time.time()
        print('Loading data...')
        source_image_list, source_label_list = load_all_images(
            source_image_filelist, source_label_filelist, scale=self.scale)
        target_image_list, target_label_list = load_all_images(
            target_image_filelist, target_label_filelist, scale=self.scale)

        mix_image_list = source_image_list + target_image_list
        mix_label_list = source_label_list + target_label_list
        mix_domain_list = source_domain_list + target_domain_list
        print(f'Data ({len(source_image_list)},{len(target_image_list)}) loading time: {time.time() - start_time}')

        if not os.path.exists('loss/'):
            os.makedirs('loss/')
        line_buffer = 1
        with open(file='loss/loss_{}.txt'.format(self.parameter['name']), mode='w', buffering=line_buffer) as loss_log:
            loss_log.write('[Train Mode]\n')
            loss_log.write(write_json(self.parameter))
            loss_log.write('\n')

            for iteration in range(self.iteration):
                # observe dice loss first
                dice_coefficient = 0.1
                discriminative_ratio = self.compute_ratio(iteration)
                coefficient = np.array([dice_coefficient, discriminative_ratio], dtype=np.float32)

                dis_only = (iteration % 100 >= 50) and (iteration >= 1000)
                seg_only = not dis_only
                if seg_only:
                    self.train_task(source_image_list, source_label_list, source_domain_list,
                                    seg_optimizer, coefficient, loss_log, iteration, phase='Segmentation')
                else:
                    self.train_task(mix_image_list, mix_label_list, mix_domain_list,
                                    dis_optimizer, coefficient, loss_log, iteration, phase='Discrimination')
                # save and test module
                if np.mod(iteration + 1, self.parameter['save_interval']) == 0:
                    self.save_checkpoint(self.parameter['checkpoint_dir'],
                                         self.parameter['model_name'], global_step=iteration + 1)
                    print('[Save] Model saved with iteration %d' % (iteration + 1))
                if np.mod(iteration + 1, self.parameter['test_interval']) == 0:
                    # test at train
                    parameter_dict = dict()
                    parameter_dict['test_image_list'] = source_image_list
                    parameter_dict['test_label_list'] = source_label_list
                    parameter_dict['test_domain_list'] = source_domain_list
                    self.test(reload=False, parameter_dict=parameter_dict)

    def compute_ratio(self, iteration):
        independent_iter = 5000
        max_ratio = 0.1
        if iteration < independent_iter:
            domain_ratio = 0.0
        else:
            domain_ratio = max_ratio * (iteration - independent_iter) / (self.iteration - independent_iter)
        # Todo: still need further adjustment
        return domain_ratio

    def train_task(self, train_image_list, train_label_list, train_domain_list,
                   optimizer, coefficient, loss_log, iteration, phase):
        start_time = time.time()
        image_batch, label_batch, domain_batch = load_train_batches(
            train_image_list, train_label_list, train_domain_list, self.input_size,
            self.batch_size, flipping=self.augmentation, rotation=self.augmentation)
        # update network

        print(f'Data loading time: {time.time() - start_time}')

        _, seg_entropy, seg_dice, seg_loss, dis_loss, domain_prob = self.session.run(
            [optimizer, self.seg_entropy, self.seg_dice, self.seg_loss, self.dis_loss, self.domain_prob],
            feed_dict={self.inputs: image_batch, self.label: label_batch,
                       self.domain: domain_batch, self.coefficient: coefficient})

        string_format = f'[label] {str(np.unique(label_batch))} [Domain] {str(domain_batch)} [Phase] {phase}\n'
        string_format += f'[Iteration] {iteration + 1} time: {time.time() - start_time:.{4}} ' \
                         f'[Loss] seg_entropy: {seg_entropy:.{8}} seg_dice: {seg_dice:.{8}}\n' \
                         f'seg_loss: {seg_loss:.{8}} dis_loss: {dis_loss:.{8}} [Classify] {domain_prob}\n\n'

        loss_log.write(string_format)
        print(string_format, end='')

    def test(self, reload=True, parameter_dict=None):
        # reload model or image
        if reload:
            # self.session.run(tf.global_variables_initializer())
            if self.load_checkpoint(self.parameter['checkpoint_dir']):
                print(" [*] Load Success")
            else:
                print(" [!] Load Failed")
                exit(1)  # exit with load error

            test_image_filelist, test_label_filelist = generate_filelist(
                self.parameter['test_data_dir'], self.parameter['test_label_dir'])

            # load all images to save time
            start_time = time.time()
            print('Loading data...')
            test_image_list, test_label_list = load_all_images(
                test_image_filelist, test_label_filelist, scale=self.scale)
            # 0: source-ct, 1: target-mri
            test_domain_list = [0] * len(test_label_filelist)
            print(f'Data loading time: {time.time() - start_time}')
        else:
            # skip test if error
            if parameter_dict is None:
                return
            test_image_list = parameter_dict['test_image_list']
            test_label_list = parameter_dict['test_label_list']
            test_domain_list = parameter_dict['test_domain_list']

        # save log
        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        # _ = tf.summary.FileWriter(logdir='logs/', graph=self.session.graph)

        if not os.path.exists('test/'):
            os.makedirs('test/')
        line_buffer = 1
        with open(file='test/test_{}.txt'.format(self.parameter['name']), mode='w', buffering=line_buffer) as loss_log:
            loss_log.write('[Test Mode]\n')
            loss_log.write(write_json(self.parameter))
            loss_log.write('\n')

            evaluation = Evaluation()
            for ith in range(len(test_label_list)):
                # not used in test
                dice_coefficient = 0.1
                discriminative_ratio = 0.1
                coefficient = np.array([dice_coefficient, discriminative_ratio], dtype=np.float32)

                i_inference = infer(image=test_image_list[ith], label=test_label_list[ith],
                                    domain=test_domain_list[ith], input_size=self.input_size,
                                    stride=self.test_stride, infer_task=self.test_task,
                                    coefficient=coefficient, loss_log=loss_log, evaluation=evaluation, sample=ith)
                i_inference = i_inference[0, :, :, :]
                np.savez('test/infer_{}_sample_{}.npz'.format(self.parameter['name'], ith), inference=i_inference)
            performance = evaluation.retrieve()
            domain_accuracy = evaluation.retrieve_domain()
            np.savez('test/test_{}.npz'.format(self.parameter['name']),
                     performance=performance, domain_accuracy=domain_accuracy)
            string_format = f'{str(performance)}\n{str(domain_accuracy)}'
            loss_log.write(string_format)
            print(string_format, end='')

    def test_task(self, image_batch, label_batch, domain_batch, coefficient,
                  loss_log, fetch_d, fetch_h, fetch_w, sample):

        start_time = time.time()
        predicted_label, predicted_domain, seg_entropy, seg_dice, seg_loss, dis_loss = self.session.run(
            [self.predicted_label, self.predicted_domain,
             self.seg_entropy, self.seg_dice, self.seg_loss, self.dis_loss],
            feed_dict={self.inputs: image_batch, self.label: label_batch,
                       self.domain: domain_batch, self.coefficient: coefficient})

        string_format = f'[Sample] {sample} [label] {str(np.unique(label_batch))} [Domain] {str(domain_batch)} ' \
                        f'd: {fetch_d:.{2}} h: {fetch_h:.{2}} w: {fetch_w:.{2}}\n'
        string_format += f'time: {time.time() - start_time:.{4}} ' \
                         f'[Loss] seg_entropy: {seg_entropy:.{8}} seg_dice: {seg_dice:.{8}}\n' \
                         f'seg_loss: {seg_loss:.{8}} dis_loss: {dis_loss:.{8}}\n\n'
        loss_log.write(string_format)
        print(string_format, end='')

        return predicted_label, predicted_domain

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
        # inconsistent batch problem
        train_batch_size = self.parameter['train_batch_size']
        model_dir = 'model_{}_{}_{}'.format(self.feature_size, train_batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoint_name))
            return True
        else:
            return False
