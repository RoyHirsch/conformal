import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], './conformal_classification/'))

import torch
import torch.nn as nn
import torchvision
import numpy as np
import pickle
from tqdm import tqdm


def get_embeds_logits(model, loader, device):
    if isinstance(model, torchvision.models.resnet.ResNet):
        embed_model = EmbedResnet(model)
    else:
        raise ValueError(f'Unsupported model type {type(model)}')

    embed_model = embed_model.to(device)
    embed_model.eval()
        
    all_embeds = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch[0].to(device)
            embeds = embed_model(x)
            preds = embed_model.fc(embeds)

            embeds = embeds.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            all_embeds.append(embeds)
            all_preds.append(preds)
            all_labels.append(batch[1].numpy())

    all_embeds = np.concatenate(all_embeds, 0)
    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)
    return all_embeds, all_preds, all_labels


def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


class EmbedResnet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def fc(self, x):
        return self.model.fc(x)


def get_model(model_name):
    if model_name == 'resnet18':
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        preprocess = weights.transforms()
        model = torchvision.models.resnet18(weights=weights)

    elif model_name == 'resnet34':
        weights = torchvision.models.ResNet34_Weights.DEFAULT
        preprocess = weights.transforms()
        model = torchvision.models.resnet34(weights=weights)

    elif model_name == 'resnet50':
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        model = torchvision.models.resnet50(weights=weights)

    elif model_name == 'resnet152':
        weights = torchvision.models.ResNet152_Weights.DEFAULT
        preprocess = weights.transforms()
        model = torchvision.models.resnet152(weights=weights)
    else:
        raise NotImplementedError
    return model, preprocess


if __name__ == "__main__":
    ###############
    # PARAMS
    ###############
    # imagenet data downloaded from:
    # https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

    model_name = 'resnet34'
    out_dir = '/home/royhirsch/conformal/data/embeds_n_logits/aug/imnet1k_r34'
    out_file_name = '200k_train.pickle'
    data_dir =  '/home/royhirsch/datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train'
    limit = 200000

    device = torch.device('cuda:1')
    batch_size = 256
    num_workers = 4

    ###############
    # MAIN
    ###############

    model, transform = get_model(model_name)
    model = model.to(device)
    model.eval()

    dataset = torchvision.datasets.ImageFolder(data_dir, transform)
    print('Found {} images'.format(len(dataset)))
    if limit:
        n_classes = len(dataset.classes)
        assert limit / n_classes == limit // n_classes
        n_per_class = int(limit / n_classes)

        label2inds = {i: [] for i in range(n_classes)} 
        for i, l in enumerate(dataset.targets):
            label2inds[l].append(i)

        filtered_samples = []
        for i in label2inds.values():
            ids = np.random.choice(i, n_per_class, replace=False).tolist()
            for j in ids:
                filtered_samples.append(dataset.samples[j])
        
        dataset.samples = filtered_samples
        print('Limited dataset has {} images'.format(len(dataset)))

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=True,
                                              pin_memory=False)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_embeds, all_preds, all_labels = get_embeds_logits(model, data_loader, device)
    print(f'Embeds shape : {all_embeds.shape}')
    print(f'Preds shape : {all_preds.shape}')
    counts = np.bincount(all_labels)
    print('Labels count: mean: {:.3f} max: {:.3f} min: {:.3f}'.format(counts.mean(),
                                                                      counts.max(),
                                                                      counts.min()))

    print(f'Acc: {(all_preds.argmax(1) == all_labels).mean()}')
    save_pickle({'embeds': all_embeds,
                 'preds': all_preds,
                 'labels': all_labels}, os.path.join(out_dir, out_file_name))
