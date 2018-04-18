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
        self.ground_truth = None
        self.coefficient = None
        self.label = None
        self.generative = None
        self.prob = None
        self.softmax_prob = None
        self.dis_loss = None
        self.entropy = None
        self.error = None
        self.gradient = None
        self.gen_loss = None
        self.trainable_variables = None
        self.gen_variables = None
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
        self.input_channels = parameter['input_channels']
        self.feature_size = parameter['feature_size']
        self.sample_from = parameter['sample_from']
        self.sample_to = parameter['sample_to']
        self.phase = parameter['phase']
        self.augmentation = parameter['augmentation']
        self.select_samples = parameter['select_samples']
        self.iteration = self.parameter['iteration']
        # Todo: pay attention to priority of sample selection

        self.build_model()

    def model(self, inputs, ground_truth):
        is_training = (self.phase == 'train')

        with tf.device(device_name_or_function=self.device[0]):
            with tf.variable_scope('gen'):
                conv_1 = conv_bn_relu(inputs=inputs, output_channels=32, kernel_size=9, stride=1,
                                      is_training=is_training, name='conv_1', use_bias=True)
                conv_2 = conv_bn_relu(inputs=conv_1, output_channels=32, kernel_size=3, stride=1,
                                      is_training=is_training, name='conv_2', use_bias=True)
                conv_3 = conv_bn_relu(inputs=conv_2, output_channels=32, kernel_size=3, stride=1,
                                      is_training=is_training, name='conv_3', use_bias=True)
                conv_4 = conv_bn_relu(inputs=conv_3, output_channels=64, kernel_size=3, stride=1,
                                      is_training=is_training, name='conv_4', use_bias=True)
                conv_5 = conv_bn_relu(inputs=conv_4, output_channels=64, kernel_size=9, stride=1,
                                      is_training=is_training, name='conv_5', use_bias=True)
                conv_6 = conv_bn_relu(inputs=conv_5, output_channels=64, kernel_size=3, stride=1,
                                      is_training=is_training, name='conv_6', use_bias=True)
                conv_7 = conv_bn_relu(inputs=conv_6, output_channels=32, kernel_size=3, stride=1,
                                      is_training=is_training, name='conv_7', use_bias=True)
                conv_8 = conv_bn_relu(inputs=conv_7, output_channels=32, kernel_size=7, stride=1,
                                      is_training=is_training, name='conv_8', use_bias=True)
                conv_9 = conv3d(inputs=conv_8, output_channels=32, kernel_size=3, stride=1,
                                use_bias=True, name='conv_9')
                out = conv3d(inputs=conv_9, output_channels=1, kernel_size=1, stride=1,
                             use_bias=True, name='out')

        with tf.device(device_name_or_function=self.device[1]):
            with tf.variable_scope('dis'):
                # Todo: the only solution?
                concat = tf.concat([out, ground_truth], axis=0, name='concat')
                dis_1 = conv_bn_relu(inputs=concat, output_channels=32, kernel_size=5, stride=1,
                                     is_training=is_training, name='dis_1')
                pool_1 = tf.layers.max_pooling3d(inputs=dis_1, pool_size=2, strides=2, name='pool_1')
                dis_2 = conv_bn_relu(inputs=pool_1, output_channels=64, kernel_size=5, stride=1,
                                     is_training=is_training, name='dis_2')
                pool_2 = tf.layers.max_pooling3d(inputs=dis_2, pool_size=2, strides=2, name='pool_2')
                dis_3 = conv_bn_relu(inputs=pool_2, output_channels=128, kernel_size=5, stride=1,
                                     is_training=is_training, name='dis_3')
                pool_3 = tf.layers.max_pooling3d(inputs=dis_3, pool_size=2, strides=2, name='pool_3')

                dis_4 = conv3d(inputs=pool_3, output_channels=256, kernel_size=5, stride=1,
                               use_bias=True, name='dis_4')
                # reshape, notice double
                reshape = tf.reshape(tensor=dis_4, shape=[2 * self.batch_size, -1])
                full_1 = tf.contrib.layers.fully_connected(
                    inputs=reshape, num_outputs=512, scope='full_1', activation_fn=tf.nn.relu,
                    weights_regularizer=tf.contrib.slim.l2_regularizer(scale=0.0005))
                # dropout?
                full_2 = tf.contrib.layers.fully_connected(
                    inputs=full_1, num_outputs=128, scope='full_2', activation_fn=tf.nn.relu,
                    weights_regularizer=tf.contrib.slim.l2_regularizer(scale=0.0005))
                # dropout?
                prob = tf.contrib.layers.fully_connected(
                    inputs=full_2, num_outputs=2, scope='full_3', activation_fn=None,
                    weights_regularizer=tf.contrib.slim.l2_regularizer(scale=0.0005))
                softmax_prob = tf.nn.softmax(logits=prob, name='prob')

        return out, prob, softmax_prob

    def build_model(self):
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, self.input_size, self.input_size,
                                            self.input_size, self.input_channels], name='inputs')
        self.ground_truth = tf.placeholder(dtype=tf.float32,
                                           shape=[self.batch_size, self.input_size, self.input_size,
                                                  self.input_size, self.input_channels], name='ground_truth')
        self.coefficient = tf.placeholder(dtype=tf.float32, shape=[3], name='coefficient')
        self.label = tf.placeholder(dtype=tf.int32, shape=[2 * self.batch_size], name='label')

        self.generative, self.prob, self.softmax_prob = self.model(self.inputs, self.ground_truth)

        # generative: 0, real: 1
        # consider later multiplication
        self.dis_loss = discriminator_entropy(self.prob, self.label)
        self.entropy = generator_entropy(self.prob, self.label)
        self.error = reconstruction_error(self.generative, self.ground_truth)
        self.gradient = gradient_difference(self.generative, self.ground_truth)
        self.gen_loss = self.entropy * self.coefficient[0] + self.error * self.coefficient[1] + \
            self.gradient * self.coefficient[2]

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
        dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_dis, beta1=beta1_dis).minimize(
            self.dis_loss, var_list=self.dis_variables)
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_gen, beta1=beta1_gen).minimize(
            self.gen_loss, var_list=self.gen_variables)

        # if self.load_checkpoint(self.parameter['checkpoint_dir']):
        #     print(" [*] Load Success")
        # else:
        #     print(" [!] Load Failed")
        self.session.run(tf.global_variables_initializer())
        # perform initialization

        if not os.path.exists('logs/'):
            os.makedirs('logs/')

        t1_image_filelist, t2_label_filelist = generate_filelist(
            self.parameter['t1_data_dir'], self.parameter['t2_label_dir'])

        start_time = time.time()
        print('Loading data...')
        t1_image_list, t2_label_list = load_all_images(t1_image_filelist, t2_label_filelist)
        print(f'Data loading time: {time.time() - start_time}')

        if not os.path.exists('loss/'):
            os.makedirs('loss/')
        line_buffer = 1
        with open(file='loss/loss_{}.txt'.format(self.parameter['name']),
                  mode='w', buffering=line_buffer) as loss_log:
            loss_log.write('[Train Mode]\n')
            loss_log.write(write_json(self.parameter))
            loss_log.write('\n')

            for iteration in range(self.iteration):
                # coefficient control
                coefficient = self.compute_coefficient(iteration)
                coefficient = np.array([coefficient, 1, 1], dtype=np.float32)
                label = [0] * self.batch_size + [1] * self.batch_size
                label = np.array(label, dtype=np.float32)

                dis_only = (iteration >= 500) and iteration % 50 < 25
                gen_only = not dis_only
                if gen_only:
                    self.train_task(t1_image_list, t2_label_list, coefficient,
                                    gen_optimizer, loss_log, iteration, phase='Generative', label=label)
                else:
                    self.train_task(t1_image_list, t2_label_list, coefficient,
                                    dis_optimizer, loss_log, iteration, phase='Discriminator', label=label)
                # save and test module
                if np.mod(iteration + 1, self.parameter['save_interval']) == 0:
                    self.save_checkpoint(self.parameter['checkpoint_dir'],
                                         self.parameter['model_name'], global_step=iteration + 1)
                    print('[Save] Model saved with iteration %d' % (iteration + 1))
                if np.mod(iteration + 1, self.parameter['test_interval']) == 0:
                    pass

    @staticmethod
    def compute_coefficient(iteration):
        independent_iter = 2000
        # max_ratio = 1
        if iteration < independent_iter:
            ratio = 0.0
        else:
            ratio = 0.5
            # max_ratio * (iteration - independent_iter) / (self.iteration - independent_iter)
        return ratio

    def train_task(self, train_image_list, train_label_list, coefficient,
                   optimizer, loss_log, iteration, phase, label=None):
        start_time = time.time()
        inputs_batch, ground_truth_batch = load_train_batches(
            train_image_list, train_label_list, self.input_size, self.batch_size,
            flipping=self.augmentation, rotation=self.augmentation)
        # update network

        _, dis_loss, entropy, error, gradient, gen_loss, prob, softmax_prob = self.session.run(
            [optimizer, self.dis_loss, self.entropy, self.error, self.gradient, self.gen_loss,
             self.prob, self.softmax_prob],
            feed_dict={self.inputs: inputs_batch, self.ground_truth: ground_truth_batch,
                       self.coefficient: coefficient, self.label: label})
        '''output'''
        prob_output = ''
        for p in softmax_prob:
            prob_output += str(p) + ' '
        string_format = f'[Phase] {phase}\n'
        string_format += f'[Iteration] {iteration + 1} time: {time.time() - start_time:.{4}} ' \
                         f'[Loss] dis_loss: {dis_loss:.{8}} gen_loss: {gen_loss:.{8}} \n' \
                         f'[Loss] entropy: {entropy:.{8}} error: {error:.{8}} gradient: {gradient:.{8}}\n' \
                         f'[Prob] prob: {prob_output}\n\n'
        loss_log.write(string_format)
        print(string_format, end='')

    def test(self, reload=True):

        test_t1_list = None
        test_t2_list = None
        test_label_list = None

        # reload model or image
        if reload:
            if self.load_checkpoint(self.parameter['checkpoint_dir']):
                print(" [*] Load Success")
            else:
                print(" [!] Load Failed")
                exit(1)  # exit with load error

            test_t1_filelist, test_t2_filelist = generate_filelist(
                self.parameter['t1_data_dir'], self.parameter['t2_label_dir'])

            # load all images to save time
            start_time = time.time()
            print('Loading data...')
            test_t1_list, test_t2_list = load_all_images(
                test_t1_filelist, test_t2_filelist)
            # generative: 0, real: 1
            test_label_list = [0] * len(test_t2_filelist)
            print(f'Data loading time: {time.time() - start_time}')

        # save log
        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        _ = tf.summary.FileWriter(logdir='logs/', graph=self.session.graph)

        if not os.path.exists('test/'):
            os.makedirs('test/')
        line_buffer = 1
        with open(file='test/test_{}.txt'.format(self.parameter['name']), mode='w', buffering=line_buffer) as loss_log:
            loss_log.write('[Test Mode]\n')
            loss_log.write(write_json(self.parameter))
            loss_log.write('\n')

            evaluation = Evaluation()
            inference = []
            for ith in range(len(test_t2_list)):
                # not used in test
                coefficient = np.array([0, 1, 1], dtype=np.float32)

                i_inference = infer(image=test_t1_list[ith], label=test_t2_list[ith], domain=test_label_list[ith],
                                    input_size=self.input_size, strike=self.input_size // 2, infer_task=self.test_task,
                                    coefficient=coefficient, loss_log=loss_log, evaluation=evaluation, sample=ith)
                inference.append(i_inference)
            inference = np.array(inference)
            # not test
            performance = evaluation.retrieve()
            dis_accuracy = evaluation.retrieve_domain()
            np.savez('test/test_{}.npz'.format(self.parameter['name']),
                     performance=performance, domain_accuracy=dis_accuracy, inference=inference)
            string_format = f'{str(performance)}\n{str(dis_accuracy)}'
            loss_log.write(string_format)
            print(string_format, end='')

    def test_task(self, image_batch, label_batch, domain_batch, coefficient,
                  loss_log, fetch_d, fetch_h, fetch_w, sample):

        start_time = time.time()
        generative, prob, dis_loss, entropy, error, gradient, gen_loss = self.session.run(
            [self.generative, self.prob,
             self.dis_loss, self.entropy, self.error, self.gradient, self.gen_loss],
            feed_dict={self.inputs: image_batch, self.ground_truth: label_batch,
                       self.label: domain_batch, self.coefficient: coefficient})

        string_format = f'[Sample] {sample} [Domain] {str(domain_batch)} ' \
                        f'd: {fetch_d:.{2}} h: {fetch_h:.{2}} w: {fetch_w:.{2}}\n'
        string_format += f'time: {time.time() - start_time:.{4}} ' \
                         f'[Loss] dis_loss: {dis_loss:.{8}} entropy: {entropy:.{8}}\n' \
                         f'error: {error:.{8}} gradient: {gradient:.{8}} gen_loss: {gen_loss:.{8}}\n\n'
        loss_log.write(string_format)
        print(string_format, end='')

        return generative, prob

    '''
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
    '''

    def save_checkpoint(self, checkpoint_dir, model_name, global_step):
        model_dir = 'model_{}_{}'.format(self.feature_size, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # function of global_step?
        self.saver.save(self.session, os.path.join(checkpoint_dir, model_name), global_step=global_step)

    def load_checkpoint(self, checkpoint_dir):
        # inconsistent batch problem
        train_batch_size = self.parameter['train_batch_size']
        model_dir = 'model_{}_{}'.format(self.feature_size, train_batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoint_name))
            return True
        else:
            return False
