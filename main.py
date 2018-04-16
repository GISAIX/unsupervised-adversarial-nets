from init import init_parameter
from iostream import write_json
from model import AdversarialNet
import argparse
import datetime
import os
import tensorflow as tf


def main(_):
    parser = argparse.ArgumentParser(description='Processing parameter arguments.')

    parser.add_argument('-g', '--gpu', help='cuda visible devices')
    parser.add_argument('-t', '--test', action='store_true', help='test phase (default: train phase)')
    parser.add_argument('-s', '--sample', help='sample selection')
    parser.add_argument('-a', '--augmentation', action='store_true',
                        help='data augmentation including flipping and rotation')
    parser.add_argument('--iteration', help='training iterations')
    parser.add_argument('--save_interval', help='save interval')
    parser.add_argument('--test_interval', help='test interval')
    parser.add_argument('--select', help='select samples from a given list')
    parser.add_argument('--feature', help='specify feature size base')
    parser.add_argument('--memory', help='set gpu memory usage fraction')
    args = parser.parse_args()

    if args.gpu:
        gpu = args.gpu
    else:
        gpu = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if args.test:
        phase = 'test'
    else:
        phase = 'train'
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = phase + '_' + current_time

    # parameter initialization
    parameter = init_parameter(name)
    parameter['gpu'] = gpu
    parameter['phase'] = phase
    # Todo: inconsistent batch problem
    if phase == 'test':
        parameter['batch_size'] = 1
    if args.sample:
        sample_select = args.sample.strip().split(',')
        if len(sample_select) == 1:
            parameter['sample_from'] = int(sample_select[0])
            parameter['sample_to'] = int(sample_select[0])
        elif len(sample_select) == 2 and sample_select[1] != '':
            parameter['sample_from'] = int(sample_select[0])
            parameter['sample_to'] = int(sample_select[1])
        else:
            print('[!] Sample selection error.')
            exit(1)
    if args.augmentation:
        parameter['augmentation'] = True
    else:
        parameter['augmentation'] = False
    if args.iteration:
        parameter['iteration'] = int(args.iteration)
    if args.save_interval:
        parameter['save_interval'] = int(args.save_interval)
    if args.test_interval:
        parameter['test_interval'] = int(args.test_interval)
    if args.select:
        samples = list()
        string = args.select.strip().split(',')
        for var in string:
            samples.append(int(var))
        parameter['select_samples'] = samples
    else:
        parameter['select_samples'] = None
    if args.feature:
        parameter['feature_size'] = int(args.feature)
    if args.memory:
        memory = float(args.memory)
    else:
        memory = 0.475

    # json
    if not os.path.exists('json/'):
        os.makedirs('json/')
    string = write_json(parameter, file=True, filename='json/' + name + '.json')
    print(string)

    print(f'Memory fraction: {memory}')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as session:
        model = AdversarialNet(session=session, parameter=parameter)
        if parameter['phase'] == 'train':
            print('Training Phase...')
            model.train()
        if parameter['phase'] == 'test':
            print('Testing Phase...')
            model.test()


if __name__ == '__main__':
    tf.app.run()
