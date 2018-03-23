

def init_parameter(name):
    parameter = dict()
    # shape setting
    parameter['batch_size'] = 1
    parameter['input_size'] = 64
    parameter['input_channels'] = 1
    parameter['output_size'] = parameter['input_size']
    parameter['output_class'] = 8
    # learning strategy
    parameter['learning_rate'] = 0.01
    parameter['beta1'] = 0.5
    parameter['epoch'] = 35000
    # data location
    parameter['source_data_dir'] = '../MM-WHS/ct_train1/'
    parameter['source_label_dir'] = '../MM-WHS/ct_train1/'
    parameter['target_data_dir'] = '../MM-WHS/ct_train2/'
    parameter['target_label_dir'] = '../MM-WHS/ct_train2/'
    parameter['test_data_dir'] = '../MM-WHS/ct_test1/'
    parameter['predict_label_dir'] = '../MM-WHS/ct_test1/prediction/'
    # model configuration
    parameter['name'] = name
    parameter['model_name'] = f'mm-whs_{name}.model'
    parameter['checkpoint_dir'] = 'checkpoint/'
    parameter['scale'] = 1
    parameter['test_stride'] = 32  # for overlap
    parameter['save_interval'] = 10000
    parameter['test_interval'] = 2000
    # scalable number of feature maps: default 32
    parameter['feature_size'] = 16
    # sample selection
    parameter['sample_from'] = 0
    parameter['sample_to'] = 9
    return parameter
