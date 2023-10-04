import os
import copy
import random
from tqdm import tqdm
from itertools import product
import numpy as np
import pandas as pd

from experiment import run_experiment
import utils as utils
from config import get_config_by_name


class ExperimentsLogger():
    def __init__(self, save_dir, file_name='scan_res.csv',
                 sort_field='cp_net_post_sizes', periodic_save=5) -> None:
        self.file_name = os.path.join(save_dir, file_name)
        self.sort_field = sort_field
        self.periodic_save = periodic_save
        self.df = pd.DataFrame(
            columns=['num', 'config', 'train_sizes', 'train_acc','val_sizes', 'val_acc',
                     'cp_net_sizes', 'cp_net_acc', 'cp_net_post_sizes', 'cp_net_post_acc'])
        self.i = 1

    def update(self, config, outputs):
        self.df.loc[len(self.df)] = [self.i,
                                     str(vars(config)), 
                                     outputs['train_mets']['size_mean'],
                                     outputs['train_mets']['acc'],
                                     outputs['val_mets']['size_mean'],
                                     outputs['val_mets']['acc'],
                                     outputs['test_mets']['size_mean'],
                                     outputs['test_mets']['acc'],
                                     outputs['calibrated_test_mets']['size_mean'],
                                     outputs['calibrated_test_mets']['acc']]
        self.i += 1
        if self.i % self.periodic_save == 0:
            self.save()

    def save(self, ):
        self.df = self.df.sort_values(by=self.sort_field, ascending=False)
        self.df.to_csv(self.file_name)
        print('Best exp:')
        best_exp = self.df.iloc[0]
        print(best_exp['config'])
        print('Post-test acc: {:.4f} and size: {:.2f}'.format(
            best_exp['cp_net_post_acc'],
            best_exp['cp_net_post_sizes']))

    
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
        outputs = run_experiment(modified_config)
        logger.update(modified_config, outputs)
        print('Finish exp: {}/{}'.format(i , num))

    logger.save()


def random_search(base_config, scan_params, num=3):
    logger = ExperimentsLogger(base_config.exp_dir)

    for i in tqdm(range(num)):
        params_to_modify = {}
        for k, v in scan_params.items():
            params_to_modify[k] = random.choice(v)
        modified_config = modify_config(base_config, params_to_modify)
        outputs = run_experiment(modified_config)
        logger.update(modified_config, outputs)
        print('Finish exp: {}/{}'.format(i , num))

    logger.save()


if __name__ == '__main__':
    scan_params = {'lr': [1e-3, 5e-4, 1e-4],
                   'wd': [1e-6],
                   'hidden_dim': [32, 128],
                   'drop_rate': [0, 0.3, 0.5],
                   'num_epochs': [70, 100, 120]}
    
    dataset_name= 'tissuemnist'
    name = f'{dataset_name}_alpha=0.05_10r_0310'
    out_dir = f'/home/royhirsch/conformal/exps/{dataset_name}'
    gpu_num = 0

    config = get_config_by_name(dataset_name)
    config.name = name
    config.out_dir = out_dir
    config.gpu_num = gpu_num
    config.exp_dir = os.path.join(config.out_dir, name)
    config.dump_log = True

    utils.create_logger(config.exp_dir, True)
    if not os.path.exists(config.exp_dir):
        os.makedirs(config.exp_dir)

    random_search(config, scan_params, num=60)
    # grid_search(config, scan_params)