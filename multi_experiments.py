import os
import copy
import random
from tqdm import tqdm
from itertools import product
import numpy as np
import pandas as pd


from experiment import get_config, run_experiment
import utils as utils


class ExperimentsLogger():
    def __init__(self, save_dir, file_name='scan_res.csv',
                 sort_field='val_sizes', periodic_save=5) -> None:
        self.file_name = os.path.join(save_dir, file_name)
        self.sort_field = sort_field
        self.periodic_save = periodic_save
        self.df = pd.DataFrame(
            columns=['num', 'config', 'train_acc', 'train_sizes', 'val_acc', 'val_sizes'])
        self.i = 1

    def update(self, config, val_mets, train_mets):
        self.df.loc[len(self.df)] = [self.i, str(vars(config)), 
                                     train_mets['acc'], train_mets['set_size_mean'],
                                     val_mets['acc'], val_mets['set_size_mean']]
        self.i += 1
        if self.i % self.periodic_save == 0:
            self.save()

    def save(self, ):
        self.df = self.df.sort_values(by=self.sort_field, ascending=False)
        self.df.to_csv(self.file_name)
        print('Best exp:')
        best_exp = self.df.iloc[0]
        print(best_exp['config'])
        print('Val acc: {:.4f} val size: {:.2f}'.format(best_exp['val_acc'],
                                                        best_exp['val_sizes']))

    
        
def modify_config(base_config, params_to_modify):
    config = copy.deepcopy(base_config)
    for k, v in params_to_modify.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config


def grid_search(base_config, scan_params):
    grid_params = [dict(zip(scan_params, v)) for v in product(*scan_params.values())]
    num = len(grid_params)
    print('Start scanning {} configs'.format(num))
    logger = ExperimentsLogger(base_config.exp_dir)

    for i in tqdm(range(num)):
        params_to_modify = grid_params[i]
        modified_config = modify_config(base_config, params_to_modify)
        history, val_predict_out, train_predict_out, val_mets, train_mets = run_experiment(modified_config)
        logger.update(modified_config, val_mets, train_mets)
        print('Finish exp: {}/{}'.format(i , num))

    logger.save()


def random_search(base_config, scan_params, num=3):
    logger = ExperimentsLogger(base_config.exp_dir)

    for i in tqdm(range(num)):
        params_to_modify = {}
        for k, v in scan_params.items():
            params_to_modify[k] = random.choice(v)
        modified_config = modify_config(base_config, params_to_modify)
        history, val_predict_out, train_predict_out, val_mets, train_mets = run_experiment(modified_config)
        logger.update(modified_config, val_mets, train_mets)
        print('Finish exp: {}/{}'.format(i , num))

    logger.save()


if __name__ == '__main__':
    scan_params = {'lr': [1e-3, 5e-4, 3e-4, 1e-4],
                   'wd': [1e-3, 1e-4, 1e-5],
                   'hidden_dim': [None, 512, 256, 128],
                   'drop_rate': [0, 0.3, 0.5],
                   'scheduler_name': ['none', 'plateau'],
                   'optimizer_name': ['adamw', 'sgd'],
                   'num_epochs': [60],
                   'norm': [False, True]}
    
    # scan_params ={'lr': [1e-3, 5e-4],
    #                'num_epochs': [2],
    #                'name': ['base_scan_2207']}
    
    name = 'base_scan_v2_2207'
    gpu_num = 0

    config = get_config()
    config.name = name
    config.gpu_num = gpu_num
    config.exp_dir = os.path.join(config.out_dir, name)
    utils.create_logger(config.exp_dir, False)
    if not os.path.exists(config.exp_dir):
        os.makedirs(config.exp_dir)

    random_search(config, scan_params, num=200)
    # grid_search(config, scan_params)