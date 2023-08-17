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


def get_dataloaders(config, conformal_module):
    data = load_pickle(config.file_name)
    valid_data, train_data = split_data(data, config.num_train, seed=config.seed)
    logging.info('Train/Calib shape: {} | Val shape: {}'.format(
        train_data['labels'].shape, valid_data['labels'].shape))

    if config.plat_scaling:
        train_dataloader = get_logits_dataloader(train_data['preds'],
                                                 train_data['labels'])
        t = platt_logits(train_dataloader)
        logging.info('Temp is {:.4f}'.format(t))
    else: 
        t = 1.

    train_data['probs'] = softmax(train_data['preds'] / t, 1)
    train_scores = conformal_module.get_scores(train_data['probs'],
                                               train_data['labels'])
    train_dataloader = get_dataloader(train_data['embeds'],
                                      train_data['probs'], 
                                      train_data['labels'],
                                      np.asarray(train_scores),
                                      batch_size=config.batch_size,
                                      shuffle=True, 
                                      pin_memory=True)

    valid_data['probs'] = softmax(valid_data['preds'] / t, 1)
    valid_scores = conformal_module.get_scores(valid_data['probs'], 
                                               valid_data['labels'])
    valid_dataloader = get_dataloader(valid_data['embeds'],
                                      valid_data['probs'],
                                      valid_data['labels'], 
                                      np.asarray(valid_scores), 
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      pin_memory=True)

    return train_dataloader, valid_dataloader, t