import os
import torch
from ml_collections import config_dict


def get_config_tissuemnist():
    
    cfg = config_dict.ConfigDict()

    # experiment
    cfg.name = 'tmp'
    cfg.out_dir = '/home/royhirsch/conformal/exps/tissuemnist'
    cfg.exp_dir = os.path.join(cfg.out_dir, cfg.name)

    cfg.dump_log = True
    cfg.comments = ''
    cfg.gpu_num = 1
    cfg.device = torch.device('cuda:{}'.format(cfg.gpu_num) if torch.cuda.is_available() else 'cpu')
    cfg.seed = 42

    # data
    # cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/imnet1k_r152/valid.pickle'
    cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/aug/medmnist/tissuemnist_test.pickle'
    # cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/aug/imnet1k_r152/100k_train.pickle'
    cfg.label_transform_name = 'none'
    cfg.num_train = 40000
    cfg.num_valid = 3500
    cfg.num_test = 3500
    cfg.batch_size = 128
    cfg.num_workers = 4
    cfg.pin_memory = True
    
    # conformal
    cfg.alpha = 0.1
    cfg.plat_scaling = True
    cfg.conformal_module_name = 'aps'
    cfg.use_score_clipping = True

    # model
    cfg.input_dim = 2048
    cfg.norm = False
    cfg.drop_rate = 0.0
    cfg.hidden_dim = 512

    # optim
    cfg.optimizer_name = 'adamw'
    cfg.scheduler_name = 'none'
    cfg.criteria_name = 'mse'
    cfg.lr = 5e-4
    cfg.wd = 1e-6

    # train
    cfg.num_epochs = 100
    cfg.val_interval = 5
    cfg.save_interval = 200
    cfg.monitor_met_name = 'val_loss'
    
    return cfg


def get_config_organamnist():
    
    cfg = config_dict.ConfigDict()

    # experiment
    cfg.name = 'tmp'
    cfg.out_dir = '/home/royhirsch/conformal/exps/organamnist'
    cfg.exp_dir = os.path.join(cfg.out_dir, cfg.name)

    cfg.dump_log = True
    cfg.comments = ''
    cfg.gpu_num = 1
    cfg.device = torch.device('cuda:{}'.format(cfg.gpu_num) if torch.cuda.is_available() else 'cpu')
    cfg.seed = 42

    # data
    cfg.file_name = '/home/royhirsch/conformal/data/embeds_n_logits/aug/medmnist/organmnist_test.pickle'
    cfg.label_transform_name = 'none'
    cfg.num_train = 19469
    cfg.num_valid = 2400
    cfg.num_test = 2400
    cfg.batch_size = 128
    cfg.num_workers = 4
    cfg.pin_memory = True
    
    # conformal
    cfg.alpha = 0.1
    cfg.plat_scaling = True
    cfg.conformal_module_name = 'aps'
    cfg.use_score_clipping = True

    # model
    cfg.input_dim = 2048
    cfg.norm = False
    cfg.drop_rate = 0.0
    cfg.hidden_dim = 32

    # optim
    cfg.optimizer_name = 'adamw'
    cfg.scheduler_name = 'none'
    cfg.criteria_name = 'mse'
    cfg.lr = 5e-4
    cfg.wd = 1e-6

    # train
    cfg.num_epochs = 100
    cfg.val_interval = 5
    cfg.save_interval = 200
    cfg.monitor_met_name = 'val_loss'
    
    return cfg


def get_config_by_name(name):
    if name == 'tissuemnist':
        return get_config_tissuemnist()
    elif name == 'organamnist':
        return get_config_organamnist()
    else:
        raise ValueError