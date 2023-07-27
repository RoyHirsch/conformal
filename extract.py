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
    model_name = 'resnet18'
    out_dir = '/home/royhirsch/conformal/data/embeds_n_logits/imnet1k_r18'

    device = torch.device('cuda:0')
    batch_size = 128
    num_workers = 4
    data_dir = '/home/royhirsch/conformal/conformal_classification/imagenet_val'


    ###############
    # MAIN
    ###############

    model, transform = get_model(model_name)
    model = model.to(device)
    model.eval()

    valid_dataset = torchvision.datasets.ImageFolder(data_dir, 
                                                     transform)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=False,
                                               pin_memory=True)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_embeds, all_preds, all_labels = get_embeds_logits(model, valid_loader, device)
    print(f'Embeds shape : {all_embeds.shape}')
    print(f'Preds shape : {all_preds.shape}')
    print(f'Acc: {(all_preds.argmax(1) == all_labels).mean()}')
    save_pickle({'embeds': all_embeds,
                 'preds': all_preds,
                 'labels': all_labels}, os.path.join(out_dir, 'valid.pickle'))
