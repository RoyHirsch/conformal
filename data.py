import logging
import numpy as np
import torch

from evaluate import load_pickle, split_data
from conf_tools import get_logits_dataloader, platt_logits


class ScoresDataset(torch.utils.data.Dataset):
    def __init__(self, embeds, cls_logits, cls_labels, scores):
        self.embeds = embeds
        self.cls_logits = cls_logits
        self.cls_labels = cls_labels
        self.scores = scores
        

    def __getitem__(self, i):
        return {'embeds': self.embeds[i],
                'logits': self.cls_logits[i],
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

    if config.plat_scaling:
        train_dataloader = get_logits_dataloader(train_data['preds'],
                                                 train_data['labels'])
        t = platt_logits(train_dataloader)
        logging.info('Temp is {:.4f}'.format(t))
    else: 
        t = 1.

    train_scores = conformal_module.get_scores(train_data['preds'],
                                               train_data['labels'],
                                               t)
    train_dataloader = get_dataloader(train_data['embeds'], train_data['preds'], 
                                      train_data['labels'], np.asarray(train_scores),
                                      batch_size=config.batch_size, shuffle=True, 
                                      pin_memory=True)

    valid_scores = conformal_module.get_scores(valid_data['preds'], 
                                               valid_data['labels'],
                                               t)
    valid_dataloader = get_dataloader(valid_data['embeds'], valid_data['preds'],
                                      valid_data['labels'], np.asarray(valid_scores), 
                                      batch_size=config.batch_size, shuffle=False,
                                      pin_memory=True)

    return train_dataloader, valid_dataloader, t