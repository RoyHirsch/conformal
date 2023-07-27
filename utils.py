import os
import logging
import time
import numpy as np
import torch
import random
import json
from matplotlib import pyplot as plt
from datetime import timedelta


def log(prefix, mets):
        s = ' | '.join([f'{k}: {v:.3f}' for k, v in mets.items()])
        logging.info(prefix + ': ' + s)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(log_dir, dump=True):
    filepath = os.path.join(log_dir, 'net_launcher_log.log')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # # Safety check
    # if os.path.exists(filepath) and opt.checkpoint == "":
    #     logging.warning("Experiment already exists!")

    # Create logger
    log_formatter = LogFormatter()

    if dump:
        # create file handler and set level to info
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to info
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if dump:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info('Created main log at ' + str(filepath))
    return logger


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def set_initial_random_seed(random_seed):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


class MetricLogger():
    def __init__(self):
        self.reset()

    def reset(self):
        self._losses = []
        raise NotImplementedError
    
    def update(self, preds, labels):
        raise NotImplementedError

    def update_loss(self, loss):
        self._losses.append(loss)        

    def calc(self):
        m = {'loss': np.mean(self._losses)}
        raise NotImplementedError


class RegressionMetricLogger(MetricLogger):
    def __init__(self, t=1.):
        super().__init__()
        self.t = t

    def reset(self):
        self._losses = []
        self._l2s = []
        self._set_sizes = []
        self._hits = []

    def update(self, pred_scores, true_scores, preds, labels):
        pred_scores = pred_scores.squeeze().detach().cpu()
        true_scores = true_scores.detach().cpu()
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        self._l2s += (torch.abs(pred_scores - true_scores)**2).numpy().tolist()

        # pred_scores = pred_scores.numpy()
        # true_scores = true_scores.numpy()
        # preds = preds.numpy()
        # labels = labels.numpy()

        # logits = softmax(preds / self.t, 1)
        # for s, p, l in zip(true_scores, logits, labels):
        #     sorted_p = np.sort(p)[::-1]
        #     argsort_p = np.argsort(p)[::-1]
        #     cumsum_p = np.cumsum(sorted_p)
        #     inds = np.where(cumsum_p >= s)[0]
        #     if len(inds):
        #         self._set_sizes.append(inds[0] + 1)
        #         if l in argsort_p[:int(inds[0] + 1)]:
        #             self._hits.append(1)
        #         else:
        #             self._hits.append(0)
        #     else:
        #         print('E')

    def calc(self):
        return {'loss': np.mean(self._losses),
                'l2':  np.mean(self._l2s)}
                # 'size':  np.mean(np.asarray(self._set_sizes)),
                # 'acc':  np.mean(np.asarray(self._hits))}


def pprint_mets(mets_dict):
    for k, v in mets_dict.items():
        print('{} : {:.3f}'.format(k, v))


class ModelCheckpoint():
    """ Save the model after every epoch. """

    def __init__(self, log_dir, save_best_only=False, mode='auto', monitor='val_loss', save_interval=1):

        self.log_dir = log_dir
        self.save_best_only = save_best_only
        self.save_interval = save_interval
        self.epochs_since_last_save = 0
        self.monitor = monitor

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def save_model(self, epoch, model, optimizer, mets=None):
        met = mets[self.monitor]
        self.filepath = self.log_dir + '/model.pth.tar'

        if epoch > 1 and epoch % self.save_interval == 0:
            if self.save_best_only:
                current = met
                if current is None:
                    current = '_'

                elif isinstance(current, torch.Tensor):
                    current = current.detach().cpu().numpy()

                else:
                    if self.monitor_op(current, self.best):
                        self.best = current
                        torch.save({'epoch': epoch,
                                    'metric': self.best,
                                    'state_dict': model.state_dict(),
                                    'optim_dict': optimizer.state_dict()},
                                   self.filepath)

            else:
                torch.save({'epoch': epoch,
                            'metric': met,
                            'state_dict': model.state_dict(),
                            'optim_dict': optimizer.state_dict()},
                           self.filepath)

    def early_stop_save_model(self, history, model, optimizer):
        epoch = history.curr_epoch
        filepath = self.log_dir + '/early_stop_epoch_{}_{}_{}_model.pth.tar'.format(
            epoch,
            self.monitor,
            str(round(history.get_last_value(self.monitor), 4)))

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict()},
                   filepath)