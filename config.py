import os
import torch
import copy
from ml_collections import config_dict


_PER_DATASET_CONFIG = {

    'imnet_r152': {
        'out_dir' : '/home/royhirsch/conformal/exps/cifar10',
        'dataset_name' : 'imnet_r152',
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/imnet1k_r152/valid.pickle',
        'input_dim': 2048, 
        # 'hidden_dim' : 32,
        # 'lr': 1e-4,
        # 'wd': 1e-6,
        # 'num_epochs': 140,
    },
    
    'cifar100_r56': {
        'out_dir' : '/home/royhirsch/conformal/exps/cifar100',
        'dataset_name' : 'cifar100_r56',
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/cifar100/resnet56/val.pickle',
        'input_dim': 64, 
        'hidden_dim' : 64,
        'lr': 1e-4,
        'wd': 1e-5,
        'num_epochs': 120,
    },

    'cifar10_r56': {
        'out_dir' : '/home/royhirsch/conformal/exps/cifar10',
        'dataset_name' : 'cifar10_r56',
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/cifar10/resnet56/val.pickle',
        'input_dim': 64, 
        'hidden_dim' : 32,
        'lr': 1e-4,
        'wd': 1e-6,
        'num_epochs': 140,
    },

    'cifar10_r20': {
        'out_dir' : '/home/royhirsch/conformal/exps/cifar10',
        'dataset_name' : 'cifar10_r20',
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/cifar10/resnet20/val.pickle',
        'input_dim': 64, 
        'hidden_dim' : 32,
        'lr': 1e-4,
        'wd': 1e-6,
        'num_epochs': 100,
    },

    'tissuemnist': {
        'out_dir' : '/home/royhirsch/conformal/exps/tissuemnist',
        'dataset_name' : 'tissuemnist',
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/aug/medmnist/tissuemnist_test.pickle',
        'hidden_dim' : 512,
    },
    
    'organamnist': {
        'out_dir' : '/home/royhirsch/conformal/exps/organamnist',
        'dataset_name' : 'organamnist',
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/aug/medmnist/organmnist_test.pickle',
        'hidden_dim' : 32,
    },

    'organsmnist': {
        'out_dir' : '/home/royhirsch/conformal/exps/organsmnist',
        'dataset_name' : 'organsmnist' ,
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/aug/medmnist/organsmnist_test.pickle',
        'hidden_dim' : 8,
    },

    'organcmnist': {
        'out_dir' : '/home/royhirsch/conformal/exps/organcmnist',
        'dataset_name' : 'organcmnist' ,
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/aug/medmnist/organcmnist_test.pickle',
        'hidden_dim' : 8,
        'num_epochs': 50,
    },

    'octmnist': {
        'out_dir' : '/home/royhirsch/conformal/exps/octmnist',
        'dataset_name' : 'octmnist' ,
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/aug/medmnist/octmnist_test.pickle',
        'hidden_dim' : 8,
        'num_epochs': 50,
        'k_raps': 4,
    },

    'pathmnist': {
        'out_dir' : '/home/royhirsch/conformal/exps/pathmnist',
        'dataset_name' : 'pathmnist' ,
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/aug/medmnist/pathmnist_test.pickle',
        'hidden_dim' : 8,
        'num_epochs': 100,
    }

}


def modify_config(base_config, params_to_modify):
    config = copy.deepcopy(base_config)
    for k, v in params_to_modify.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config


def get_config():
    
    cfg = config_dict.ConfigDict()

    # experiment
    cfg.name = 'tmp'
    cfg.out_dir = ''
    cfg.exp_dir = os.path.join(cfg.out_dir, cfg.name)

    cfg.dump_log = False
    cfg.comments = ''
    cfg.gpu_num = 0
    cfg.device = torch.device('cuda:{}'.format(cfg.gpu_num) if torch.cuda.is_available() else 'cpu')
    cfg.seed = 42

    # data
    cfg.dataset_name = ''
    cfg.file_name = ''
    cfg.label_transform_name = 'none'
    cfg.par_valid = 0.08
    cfg.par_test = 0.08
    cfg.batch_size = 128
    cfg.num_workers = 4
    cfg.pin_memory = True
    
    # conformal
    cfg.alpha = 0.1
    cfg.plat_scaling = True
    cfg.conformal_module_name = 'aps'
    cfg.use_score_clipping = False
    cfg.method_name = 'add'

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
    
    # baselines
    cfg.k_raps = 5
    return cfg


def get_config_by_name(name):
    config = get_config()
    return modify_config(config, _PER_DATASET_CONFIG[name])
