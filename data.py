import logging
import numpy as np
import torch
import random
from scipy.special import softmax

from evaluate import load_pickle, split_data
from conf_tools import get_logits_dataloader, platt_logits


def interpolate(a, b, tau):
    return (a * tau) + (b * (1. - tau))


class ScoresDataset(torch.utils.data.Dataset):
    def __init__(self, embeds, cls_probs, cls_labels, scores, rand=0):
        self.embeds = embeds
        self.cls_probs = cls_probs
        self.cls_labels = cls_labels
        self.scores = scores
        self.rand = rand
        

    def __getitem__(self, i):
        # if self.rand:
        #     if random.random() < self.rand:
        #         pass
        #     else:
        #         tau = random.random()
        #         j = random.choice(range(self.__len__()))
        #         return {'embeds': interpolate(self.embeds[i], self.embeds[j], tau),
        #                 'probs': interpolate(self.cls_probs[i], self.cls_probs[j], tau),
        #                 'labels': interpolate(self.cls_labels[i], self.cls_labels[j], tau),
        #                 'scores': interpolate(self.scores[i], self.scores[j], tau)}

        return {'embeds': self.embeds[i],
                'probs': self.cls_probs[i],
                'labels': self.cls_labels[i],
                'scores': self.scores[i]}

    def __len__(self):
        return self.cls_labels.size(0)


def get_dataloader(embeds, cls_logits, cls_labels, scores,
                   batch_size=128, shuffle=False,
                   pin_memory=True):
    dataset = ScoresDataset(
        torch.from_numpy(embeds), 
        torch.from_numpy(cls_logits),
        torch.from_numpy(cls_labels).long(),
        torch.from_numpy(scores).float())
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def read_and_split(config):
    data = load_pickle(config.file_name)
    return split_data(data, config.num_train, config.num_test, config.num_valid, seed=config.seed)


def get_dataloaders(config, conformal_module, get_data_func=read_and_split):
    data = get_data_func(config)
    for k, v in data.items():
        logging.info('{} shape: {}'.format(k, v['labels'].shape))

    if config.plat_scaling:
        train_dataloader = get_logits_dataloader(data['train']['preds'],
                                                 data['train']['labels'])
        t = platt_logits(train_dataloader)
        logging.info('Temp is {:.4f}'.format(t))
    else: 
        t = 1.

    dls = {}
    for k, v in data.items():
        v['probs'] = softmax(v['preds'] / t, 1)
        scores = conformal_module.get_scores(v['probs'], v['labels'])
        dl = get_dataloader(v['embeds'], v['probs'], v['labels'],
                            np.asarray(scores),
                            batch_size=config.batch_size,
                            shuffle=True if k == 'train' else False,
                            pin_memory=True)
        dls[k] = dl

    return dls, t