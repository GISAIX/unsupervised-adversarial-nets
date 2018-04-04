

def init_parameter(name):
    parameter = dict()
    # shape setting
    parameter['batch_size'] = 2
    parameter['input_size'] = 32
    parameter['input_channels'] = 1
    parameter['output_size'] = parameter['input_size']
    parameter['output_class'] = 8
    # learning strategy
    parameter['learning_rate_dis'] = 0.001
    parameter['beta1_dis'] = 0.5
    parameter['learning_rate_seg'] = 0.001
    parameter['beta1_seg'] = 0.5
    parameter['iteration'] = 35000
    # data location
    parameter['source_data_dir'] = '../MM-WHS/ct_train/'
    parameter['source_label_dir'] = '../MM-WHS/ct_train/'
    parameter['target_data_dir'] = '../MM-WHS/mr_train/'
    parameter['target_label_dir'] = '../MM-WHS/mr_train/'
    parameter['test_source_dir'] = '../MM-WHS/ct_test/'
    parameter['test_target_dir'] = '../MM-WHS/mr_test/'
    parameter['predict_label_dir'] = '../MM-WHS/prediction/'
    # model configuration
    parameter['name'] = name
    parameter['model_name'] = f'mm-whs_{name}.model'
    parameter['checkpoint_dir'] = 'checkpoint/'
    parameter['scale'] = 1
    parameter['test_stride'] = 16  # for overlap
    parameter['save_interval'] = 5000
    parameter['test_interval'] = 2000
    # scalable number of feature maps: default 32
    parameter['feature_size'] = 32
    # sample selection
    parameter['sample_from'] = 0
    parameter['sample_to'] = 9
    # it is not used now
    return parameter
