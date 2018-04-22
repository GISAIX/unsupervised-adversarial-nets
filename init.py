

def init_parameter(name):
    parameter = dict()
    # shape setting
    parameter['batch_size'] = 6
    parameter['train_batch_size'] = parameter['batch_size']
    parameter['input_size'] = 32
    parameter['input_channels'] = 1
    parameter['output_size'] = parameter['input_size']
    parameter['output_class'] = 8
    # learning strategy
    parameter['learning_rate_dis'] = 0.00001
    parameter['beta1_dis'] = 0.5
    parameter['learning_rate_seg'] = 0.00001
    parameter['beta1_seg'] = 0.5
    parameter['iteration'] = 35000
    parameter['augmentation'] = True
    # data location
    parameter['dataset'] = 'MM-WHS'  # 'iSeg'
    parameter['source_data_dir'] = '../MM-WHS/ct_train1/'
    parameter['source_label_dir'] = '../MM-WHS/ct_train1/'
    parameter['target_data_dir'] = '../MM-WHS/mr_train1/'
    parameter['target_label_dir'] = '../MM-WHS/mr_train1/'
    parameter['test_data_dir'] = '../MM-WHS/ct_train2/'
    parameter['test_label_dir'] = '../MM-WHS/ct_train2/'
    parameter['predict_label_dir'] = '../MM-WHS/prediction/'
    # model configuration
    parameter['name'] = name
    parameter['model_name'] = f'mm-whs_{name}.model'
    parameter['checkpoint_dir'] = 'checkpoint/'
    parameter['scale'] = 1
    parameter['test_stride'] = 32  # for overlap
    parameter['save_interval'] = 5000
    parameter['test_interval'] = 2000  # template observation
    # scalable number of feature maps: default 32
    parameter['feature_size'] = 32
    # sample selection
    parameter['sample_from'] = 0
    parameter['sample_to'] = 9
    # it is not used now
    return parameter
