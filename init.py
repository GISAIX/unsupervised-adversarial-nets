

def init_parameter(name):
    parameter = dict()
    # shape setting
    # Todo: should change batch size
    parameter['batch_size'] = 10
    parameter['input_size'] = 32
    parameter['input_channels'] = 1
    # learning strategy
    parameter['learning_rate_gen'] = 1e-6
    parameter['beta1_gen'] = 0.5
    parameter['learning_rate_dis'] = 1e-6
    parameter['beta1_dis'] = 0.5
    parameter['iteration'] = 35000
    # data location
    parameter['mri_data_dir'] = '../iSeg/iSeg-2017-Training/'
    parameter['ct_label_dir'] = '../iSeg/iSeg-2017-Training/'
    # parameter['predict_label_dir'] = '../MM-WHS/prediction/'
    # model configuration
    parameter['name'] = name
    parameter['model_name'] = f'mri_ct_{name}.model'
    parameter['checkpoint_dir'] = 'checkpoint/'
    parameter['test_stride'] = 16  # for overlap
    parameter['save_interval'] = 10000
    parameter['test_interval'] = 2000
    # scalable number of feature maps: default 32
    parameter['feature_size'] = 16
    # sample selection
    parameter['sample_from'] = 0
    parameter['sample_to'] = 9
    return parameter
