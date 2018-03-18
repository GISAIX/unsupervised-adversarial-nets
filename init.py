

def init_parameter(name):
    parameter = dict()
    # shape setting
    parameter['batch_size'] = 1
    parameter['input_size'] = 96
    parameter['input_channels'] = 1
    parameter['output_size'] = parameter['input_size']
    parameter['output_class'] = 3
    # learning strategy
    parameter['learning_rate'] = 0.01
    parameter['beta1'] = 0.5
    parameter['epoch'] = 35000
    # data location
    parameter['train_data_dir'] = f'../hvsmr/crop/data/'
    parameter['train_label_dir'] = f'../hvsmr/crop/label/'
    parameter['test_data_dir'] = f'../hvsmr/crop/data/'
    parameter['predict_label_dir'] = f'../hvsmr/crop/predict/'
    # model configuration
    parameter['name'] = name
    parameter['model_name'] = f'hvsmr_crop_{name}.model'
    parameter['checkpoint_dir'] = 'checkpoint/'
    parameter['scale'] = 1
    parameter['test_stride'] = 32  # for overlap
    parameter['save_interval'] = 10000
    parameter['test_interval'] = 2000
    # scalable number of feature maps: default 32
    parameter['feature_size'] = 32
    # sample selection
    parameter['sample_from'] = 0
    parameter['sample_to'] = 9
    return parameter
