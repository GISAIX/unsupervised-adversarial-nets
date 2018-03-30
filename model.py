from conv import *
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
        self.inputs = None
        self.label = None
        self.trainable_variables = None
        self.gen_variables = None
        self.dis_variables = None
        self.saver = None
        self.coefficient = None
        self.generated_ct = None
        self.prob = None
        self.dis_loss = None
        self.gen_loss = None
        # frequently used parameters
        gpu_number = len(parameter['gpu'].split(','))
        if gpu_number > 1:
            self.device = ['/gpu:0', '/gpu:1', '/cpu:0']
        else:
            self.device = ['/gpu:0', '/gpu:0', '/cpu:0']
        self.batch_size = parameter['batch_size']
        self.input_size = parameter['input_size']
        self.input_channels = parameter['input_channels']
        self.feature_size = parameter['feature_size']
        self.sample_from = parameter['sample_from']
        self.sample_to = parameter['sample_to']
        self.phase = parameter['phase']
        self.select_samples = parameter['select_samples']
        self.iteration = self.parameter['iteration']
        # Todo: pay attention to priority of sample selection

        self.build_model()

    def model(self, inputs, label):
        is_training = (self.phase == 'train')

        with tf.device(device_name_or_function=self.device[0]):
            with tf.variable_scope('gen'):
                kernel_size = 3
                conv_1 = conv_bn_relu(inputs=inputs, output_channels=32, kernel_size=kernel_size, stride=1,
                                      is_training=is_training, name='conv_1')
                conv_2 = conv_bn_relu(inputs=conv_1, output_channels=32, kernel_size=kernel_size, stride=1,
                                      is_training=is_training, name='conv_2')
                conv_3 = conv_bn_relu(inputs=conv_2, output_channels=32, kernel_size=kernel_size, stride=1,
                                      is_training=is_training, name='conv_3')
                conv_4 = conv_bn_relu(inputs=conv_3, output_channels=64, kernel_size=kernel_size, stride=1,
                                      is_training=is_training, name='conv_4')
                conv_5 = conv_bn_relu(inputs=conv_4, output_channels=64, kernel_size=kernel_size, stride=1,
                                      is_training=is_training, name='conv_5')
                conv_6 = conv_bn_relu(inputs=conv_5, output_channels=64, kernel_size=kernel_size, stride=1,
                                      is_training=is_training, name='conv_6')
                conv_7 = conv_bn_relu(inputs=conv_6, output_channels=32, kernel_size=kernel_size, stride=1,
                                      is_training=is_training, name='conv_7')
                conv_8 = conv_bn_relu(inputs=conv_7, output_channels=32, kernel_size=kernel_size, stride=1,
                                      is_training=is_training, name='conv_8')
                out = conv3d(inputs=conv_8, output_channels=1, kernel_size=1, stride=1,
                             use_bias=True, name='out')

        with tf.device(device_name_or_function=self.device[1]):
            with tf.variable_scope('dis'):
                kernel_size = 5
                concat = tf.concat([label, out], axis=0, name='concat')
                dis_1 = conv_bn_relu(inputs=concat, output_channels=32, kernel_size=kernel_size, stride=1,
                                     is_training=is_training, name='dis_1')
                pool_1 = tf.layers.max_pooling3d(inputs=dis_1, pool_size=2, strides=2, name='pool_1')
                dis_2 = conv_bn_relu(inputs=pool_1, output_channels=64, kernel_size=kernel_size, stride=1,
                                     is_training=is_training, name='dis_2')
                pool_2 = tf.layers.max_pooling3d(inputs=dis_2, pool_size=2, strides=2, name='pool_2')
                dis_3 = conv_bn_relu(inputs=pool_2, output_channels=128, kernel_size=kernel_size, stride=1,
                                     is_training=is_training, name='dis_3')
                pool_3 = tf.layers.max_pooling3d(inputs=dis_3, pool_size=2, strides=2, name='pool_3')
                dis_4 = conv_bn_relu(inputs=pool_3, output_channels=256, kernel_size=kernel_size, stride=1,
                                     is_training=is_training, name='dis_4')
                pool_4 = tf.layers.max_pooling3d(inputs=dis_4, pool_size=2, strides=2, name='pool_4')
                average = tf.reduce_mean(input_tensor=pool_4, axis=[1, 2, 3], name='average_pooling')
                full_1 = tf.contrib.layers.fully_connected(
                    inputs=average, num_outputs=512, scope='full_1',
                    weights_regularizer=tf.contrib.slim.l2_regularizer(scale=0.0005))
                relu_1 = tf.nn.relu(features=full_1)
                full_2 = tf.contrib.layers.fully_connected(
                    inputs=relu_1, num_outputs=128, scope='full_2',
                    weights_regularizer=tf.contrib.slim.l2_regularizer(scale=0.0005))
                relu_2 = tf.nn.relu(features=full_2)
                full_3 = tf.contrib.layers.fully_connected(
                    inputs=relu_2, num_outputs=1, scope='full_3',
                    weights_regularizer=tf.contrib.slim.l2_regularizer(scale=0.0005))
                prob = tf.nn.sigmoid(x=full_3)

                print(out, prob)
                return out, prob

    def build_model(self):
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, self.input_size, self.input_size,
                                            self.input_size, self.input_channels], name='inputs')
        self.label = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, self.input_size, self.input_size,
                                            self.input_size, self.input_channels], name='label')
        self.coefficient = tf.placeholder(dtype=tf.float32, shape=[1], name='coefficient')
        self.generated_ct, self.prob = self.model(self.inputs, self.label)

        self.dis_loss = - (tf.log(self.prob[0, 0]) + tf.log(1 - self.prob[1, 0])) / 2
        self.gen_loss = - tf.log(self.prob[1, 0]) * self.coefficient + tf.reduce_mean(
            (self.generated_ct - self.label) * (self.generated_ct - self.label))

        # derivative
        self.trainable_variables = tf.trainable_variables()
        self.gen_variables = tf.trainable_variables(scope='gen')
        self.dis_variables = tf.trainable_variables(scope='dis')
        self.saver = tf.train.Saver(max_to_keep=20)
        print('Model built.')

    def train(self):
        learning_rate_gen = self.parameter['learning_rate_gen']
        beta1_gen = self.parameter['beta1_gen']
        learning_rate_dis = self.parameter['learning_rate_dis']
        beta1_dis = self.parameter['beta1_dis']

        # dynamics problem -> placeholder
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_gen, beta1=beta1_gen).minimize(
            self.gen_loss, var_list=self.gen_variables)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_dis, beta1=beta1_dis).minimize(
            self.dis_loss, var_list=self.dis_variables)

        self.session.run(tf.global_variables_initializer())

        # Todo: load pre-trained model and checkpoint

        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        # log_writer = tf.summary.FileWriter(logdir='logs/', graph=self.session.graph)

        mri_image_filelist, ct_label_filelist = generate_filelist(
            self.parameter['mri_data_dir'], self.parameter['ct_label_dir'])

        if not os.path.exists('loss/'):
            os.makedirs('loss/')
        line_buffer = 1
        with open(file='loss/loss_{}.txt'.format(self.parameter['name']),
                  mode='w', buffering=line_buffer) as loss_log:
            loss_log.write('[Train Mode]\n')
            loss_log.write(write_json(self.parameter))
            loss_log.write('\n')

            for iteration in range(self.iteration):
                coefficient = self.compute_coefficient(iteration)
                dis_only = (iteration >= 100) and (iteration + 1) % 2 == 1
                gen_only = not dis_only
                if gen_only:
                    self.train_task(mri_image_filelist, ct_label_filelist, coefficient,
                                    gen_optimizer, loss_log, iteration, phase='gen')
                else:
                    self.train_task(mri_image_filelist, ct_label_filelist, coefficient,
                                    dis_optimizer, loss_log, iteration, phase='adv')
                # save and test module
                if np.mod(iteration + 1, self.parameter['save_interval']) == 0:
                    self.save_checkpoint(self.parameter['checkpoint_dir'],
                                         self.parameter['model_name'], global_step=iteration + 1)
                    print('[Save] Model saved with iteration %d' % (iteration + 1))
                if np.mod(iteration + 1, self.parameter['test_interval']) == 0:
                    pass

    def compute_coefficient(self, iteration):
        independent_iter = 10000
        max_ratio = 1
        if iteration < independent_iter:
            domain_ratio = 0.0
        else:
            domain_ratio = max_ratio * (iteration - independent_iter) / (self.iteration - independent_iter)
        return np.array([domain_ratio], dtype=np.float32)

    def train_task(self, train_image_filelist, train_label_filelist, coefficient,
                   optimizer, loss_log, iteration, phase):
        start_time = time.time()
        image_batch, label_batch = load_train_batches(
            train_image_filelist, train_label_filelist, self.input_size, self.batch_size)
        # update network

        _, gen_loss, dis_loss = self.session.run(
            [optimizer, self.gen_loss, self.dis_loss],
            feed_dict={self.inputs: image_batch, self.label: label_batch,
                       self.coefficient: coefficient})

        '''temp'''
        string_format = f'[Phase] {phase}\n'
        string_format += '[Iteration] %d time: %4.4f [Loss] gen_loss: %.8f dis_loss: %.8f \n\n' \
                         % (iteration + 1, time.time() - start_time, gen_loss, dis_loss)
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
        model_dir = 'model_{}_{}'.format(self.feature_size, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # function of global_step?
        self.saver.save(self.session, os.path.join(checkpoint_dir, model_name), global_step=global_step)

    def load_checkpoint(self, checkpoint_dir):
        model_dir = 'model_{}_{}'.format(self.feature_size, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoint_name))
            return True
        else:
            return False
