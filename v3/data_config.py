import os
import torch
import copy
from ml_collections import config_dict


OUT_ROOT = '/home/royhirsch/conformal/data/embeds_n_logits'


DATASETS_METADATA = {
    'cifar100': {
        'file_name' : '',
        'dataset_name' : 'cifar100',
        'num_samples': 10000,
        'num_classes': 100},
    
    'cifar10': {
        'file_name' : '',
        'dataset_name' : 'cifar10',
        'num_samples': 10000,
        'num_classes': 10},

    'imagenet1k': {
        'file_name' : '',
        'dataset_name' : 'imagenet1k',
        'num_samples': 50000,
        'num_classes': 1000},
}


def get_config(name):
    dataset_name, model_name = name.split('_')
    metadata = DATASETS_METADATA[dataset_name]
    metadata['file_name'] = os.path.join(OUT_ROOT, dataset_name, model_name, f'{dataset_name}_{model_name}_val.pickle')
    return config_dict.ConfigDict(metadata)
