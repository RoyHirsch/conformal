import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], './conformal_classification/'))

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime

import medmnist
from medmnist import INFO, Evaluator
import PIL


OUT_ROOT = '/home/royhirsch/conformal/data/embeds_n_logits'
DATA_ROOT = '/home/royhirsch/conformal/data'


def get_logits(model, loader, device):
    model = model.to(device)
    model.eval()
        
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch[0].to(device)
            preds = model(x)
            preds = preds.detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch[1].numpy())

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)
    return all_preds, all_labels


def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


class EmbedCifarResnet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def fc(self, x):
        return self.model.fc(x)


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


def calc(model, loader, device, fold='Test'):
    all_embeds, all_preds, all_labels = get_logits(model, loader, device)
    all_labels = np.squeeze(all_labels)
    print(f'Embeds shape : {all_embeds.shape}')
    print(f'Preds shape : {all_preds.shape}')
    print(f'Labels shape : {all_labels.shape}')

    counts = np.bincount(all_labels)

    print(fold)
    print('Labels count: mean: {:.3f} max: {:.3f} min: {:.3f}'.format(counts.mean(),
                                                                      counts.max(),
                                                                      counts.min()))

    print(f'Acc: {(all_preds.argmax(1) == all_labels).mean()}')
    return all_embeds, all_preds, all_labels


def get_imnet_model(model_name):
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
        
    elif model_name == 'resnext101':
        weights = torchvision.models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        preprocess = weights.transforms()
        model = torchvision.models.resnext101_64x4d(weights=weights)
    
    elif model_name == 'vitb16':
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        preprocess = weights.transforms()
        model = torchvision.models.vit_b_16(weights=weights)

    elif model_name == 'vitl16':
        weights = torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
        preprocess = weights.transforms()
        model = torchvision.models.vit_l_16(weights=weights)

    else:
        raise NotImplementedError
    return model, preprocess


def main_medmnist():
    '''
    Download the dataset and models from:
    https://zenodo.org/records/7782114
    
    pathmnist, tissuemnist, organamnist
    '''
    
    dataset_name = 'pathmnist'
    device = torch.device('cuda:0')
    BATCH_SIZE = 128
    out_dir = f'home/royhirsch/conformal/data/embeds_n_logits/{dataset_name}/resnet50'
    out_file_name = f'{dataset_name}_test.pickle'
    include_val = True
    
    download = True
    info = INFO[dataset_name]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])])

    val_dataset = DataClass(split='val', transform=data_transform, download=download, size=224, mmap_mode='r')
    test_dataset = DataClass(split='test', transform=data_transform, download=download, size=224, mmap_mode='r')

    val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    num_classes = len(np.unique(val_dataset.labels))
    model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes)

    model.load_state_dict(torch.load(f'/home/royhirsch/conformal/data/medmnist_models/{dataset_name}/resnet50_224_1.pth')['net'])
    model = model.to(device)
    model = model.eval()

    all_embeds, all_preds, all_labels = calc(model, test_loader, device, fold='Test')
    if include_val:
        val_embeds, val_preds, val_labels = calc(model, val_loader, device, fold='Val')
        all_embeds = np.concatenate([all_embeds, val_embeds])
        all_preds = np.concatenate([all_preds, val_preds])
        all_labels = np.concatenate([all_labels, val_labels])
    
    print('Final:')
    print(f'Embeds shape : {all_embeds.shape}')
    print(f'Preds shape : {all_preds.shape}')
    print(f'Labels shape : {all_labels.shape}')

    save_pickle({'embeds': all_embeds,
                 'preds': all_preds,
                 'labels': all_labels}, os.path.join(out_dir, out_file_name))


def main_cifar(model_name = 'resnet56', dataset_name = 'cifar10', device_num=0):
    out_dir = os.path.join(OUT_ROOT, dataset_name, model_name)
    out_file_name = f'{dataset_name}_{model_name}_val.pickle'
    device = torch.device(f'cuda:{device_num}')

    device = torch.device(f'cuda:{device_num}')
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"{dataset_name}_{model_name}", pretrained=True)
    model = model.to(device)
    model.eval()

    if 'vit' in model_name:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif dataset_name == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:
        mean = [0.5070, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2761]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_preds, all_labels = get_logits(model, data_loader, device)
    print(f'Preds shape : {all_preds.shape}')
    counts = np.bincount(all_labels)
    print('Labels count: mean: {:.3f} max: {:.3f} min: {:.3f}'.format(counts.mean(),
                                                                      counts.max(),
                                                                      counts.min()))

    acc = (all_preds.argmax(1) == all_labels).mean()
    print('Acc: {:.3f}'.format(acc))
    save_pickle({
        'preds': all_preds,
        'labels': all_labels,
        'metadata': {
            'tiemdate': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': model_name,
            'dataset': dataset_name,
            'acc': acc,
            }
        },
                os.path.join(out_dir, out_file_name))


def main_imnet1k(model_name = 'vitl16', dataset_name = 'imagenet1k', device_num=0):
    data_root =  os.path.join(DATA_ROOT, 'imagenet-val')
    out_dir = os.path.join(OUT_ROOT, dataset_name, model_name)
    out_file_name = f'{dataset_name}_{model_name}_val.pickle'
    device = torch.device(f'cuda:{device_num}')

    model, transform = get_imnet_model(model_name)    
    dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    all_preds, all_labels = get_logits(model, data_loader, device)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f'Preds shape : {all_preds.shape}')
    counts = np.bincount(all_labels)
    print('Labels count: mean: {:.3f} max: {:.3f} min: {:.3f}'.format(counts.mean(),
                                                                      counts.max(),
                                                                      counts.min()))
    acc = (all_preds.argmax(1) == all_labels).mean()
    print('Acc: {:.3f}'.format(acc))
    save_pickle({
        'preds': all_preds,
        'labels': all_labels,
        'metadata': {
            'tiemdate': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': model_name,
            'dataset': dataset_name,
            'acc': acc,
            }
        },
                os.path.join(out_dir, out_file_name))


if __name__ == "__main__":
    # main_imnet1k()
    main_cifar()
    # main_medmnist()
    