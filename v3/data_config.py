import os
import torch
import copy
from ml_collections import config_dict


_PER_DATASET_CONFIG = {
    
    'cifar100_r56': {
        'dataset_name' : 'cifar100_r56',
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/cifar100/resnet56/val.pickle',
        'input_dim': 64, 
        'num_samples': 10000,
        'num_classes': 100,
    },

    'cifar10_r56': {
        'dataset_name' : 'cifar10_r56',
        'file_name' : '/home/royhirsch/conformal/data/embeds_n_logits/cifar10/resnet56/val.pickle',
        'input_dim': 64, 
        'num_samples': 10000,
        'num_classes': 10,


    },

}


def get_config(name):
    return config_dict.ConfigDict(_PER_DATASET_CONFIG[name])
