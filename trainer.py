import os
import numpy as np
import torch
import logging
from tqdm import tqdm
import torch.optim as optim

from utils import ModelCheckpoint


def get_optimizer(model, config):
    if config.optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    elif config.optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.wd)
    else:
        raise ValueError('Invalid optimizer {}'.format(config.optimizer_name))
    return optimizer


def get_scheduler(optimizer, config):
    if config.scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10,
        cooldown=20, min_lr=config.lr ** 0.01, verbose=True)

    elif config.scheduler_name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.)
    else:
        raise ValueError('Invalid scheduler {}'.format(config.scheduler_name))
    return scheduler


class LabelsTransform():
    def label_transform(self, labels):
        return torch.log(labels / (1. -  + 1e-8))

    def preds_transform(self, preds):
        return 1 / (np.exp(- preds) + 1)


class LambdaLabelsTransform():
    def label_transform(self, labels):
        return labels

    def preds_transform(self, preds):
        return preds


def get_label_transform(transform_name):
    logging.info(f'Label transform: {transform_name}')
    if transform_name == 'none':
        return LambdaLabelsTransform()
    elif transform_name == 'log':
        return LabelsTransform()
    else:
        raise ValueError


class Trainer():
    def __init__(self, criteria, metric_logger, config):
        self.criteria = criteria 
        self.metric_logger = metric_logger
        self.config = config
        self.transform = get_label_transform(config.label_transform_name)
        self.history = {}

    def _history_update(self, m):
        for k, v in m.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

    def to_device(self, batch, device=None):
        device = self.config.device if device == None else device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k in ['scores', 'embeds']:
                    batch[k] = v.to(self.config.device)
        return batch

    def get_label(self, batch):
        labels = batch['scores']
        return self.transform.label_transform(labels)

    def calc_loss(self, preds, labels):
        return self.criteria(preds.squeeze(), labels)
    
    def forward(self, model, batch):
        out = model(batch['embeds'])
        return out
    
    def _log_mets(self, d, epoch, mode=''):
        mets_str = ' | '.join(['{}={:.4f}'.format(k, v) for k, v in d.items()])
        logging.info(f'E{epoch}-{mode}: {mets_str}')

    def train_epoch(self, model, train_loader, optimizer):
        model.train()
        met = self.metric_logger()
        for i, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            batch = self.to_device(batch)
            labels = self.get_label(batch)
            predictions = self.forward(model, batch)
            loss = self.calc_loss(predictions, labels)

            loss.backward()
            optimizer.step()

            met.update(predictions, labels)
            met.update_loss(loss.item())
        return met.calc()

    def evaluate(self, model, data_loader):
        model.eval()
        met = self.metric_logger()
        with torch.no_grad():
            for batch in data_loader:
                batch = self.to_device(batch)
                labels = self.get_label(batch)
                predictions = self.forward(model, batch)
                loss = self.calc_loss(predictions, labels)

                met.update(predictions, labels)
                met.update_loss(loss.item())
        return met.calc()
    
    def predict(self, model, data_loader):
        model.eval()
        pred_scores = []
        true_scores = []
        cls_probs = []
        cls_labels = []
        with torch.no_grad():
            for batch in data_loader:
                batch = self.to_device(batch)
                predictions = self.forward(model, batch)

                predictions = predictions.squeeze().detach().cpu().numpy()
                predictions = self.transform.preds_transform(predictions)
                pred_scores.append(predictions)

                scores = batch['scores'].squeeze().detach().cpu().numpy()
                true_scores.append(scores)

                cls_probs.append(batch['probs'].detach().cpu().numpy())
                cls_labels.append(batch['labels'].detach().cpu().numpy())

        return {'pred_scores': np.concatenate(pred_scores),
                'true_scores': np.concatenate(true_scores),
                'cls_probs': np.concatenate(cls_probs),
                'cls_labels': np.concatenate(cls_labels)}

    def fit(self, model, train_loader, test_loader, optimizer, scheduler, valid_loader=None):
        num_epochs = self.config.num_epochs
        self.saver = ModelCheckpoint(
            log_dir=self.config.exp_dir,
            save_best_only=True,
            monitor=self.config.monitor_met_name,
            mode='auto',
            save_interval=1)
        
        logging.info('Start training for {} epochs:'.format(num_epochs))
        logging.info('Number of batches in train epoch: {}'.format(len(train_loader)))

        for epoch in range(1, num_epochs + 1):
            logging.info('E {}/{} |'.format(epoch, num_epochs))

            train_mets = self.train_epoch(model, train_loader, optimizer)
            self._log_mets(train_mets, epoch, mode='Train')
            self._history_update(train_mets)
            
            if getattr(self.config, 'val_interval', False) and epoch % self.config.val_interval == 0:
                valid_mets = self.evaluate(model, valid_loader)
                valid_mets = {f'val_{k}': v for k, v in valid_mets.items()}
                scheduler.step(valid_mets['val_loss'])
                self._log_mets(valid_mets, epoch, mode='Validation')
                self._history_update(valid_mets)
                self.saver.save_model(epoch, model, optimizer, mets=valid_mets)

        # best_checkpoint = torch.load(self.saver.filepath)
        # logging.info('Loading best checkpoint for epoch: {} | best val_loss: {:.4f}'.format(
        #     best_checkpoint['epoch'], best_checkpoint['metric']))
        # model.load_state_dict(best_checkpoint['state_dict'])

        test_mets = self.evaluate(model, test_loader)
        self._log_mets(test_mets, epoch, mode='Test')
