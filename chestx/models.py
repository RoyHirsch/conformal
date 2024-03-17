from torch import nn
from torchvision import models, transforms


def get_model(model_name):
    if model_name == 'resnet18':
        model = ResNet18(num_classes=15)
    if model_name == 'resnet50':
        model = ResNet50(num_classes=15)
    elif model_name == 'densenet121':
        model = DenseNet(num_classes=15)
    elif model_name == 'vgg_bn':
        model = VGG_BN(num_classes=15)

    return model



for_pad = lambda s: s if s > 2 else 3


class ConvBlock(nn.Module):
    def __init__(self, ni, nf, size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(ni, nf, kernel_size=size, stride=stride, padding=(for_pad(size) - 1)//2, bias=False),
                                        nn.BatchNorm2d(nf),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True)  
                                        )
        
    def forward(self, x):
        return self.conv_block(x)
    

class ConvLayer(nn.Module):
    def __init__(self, ni, nf, size=3, stride=1):
        super(ConvLayer, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(ni, nf, kernel_size=size, stride=stride, padding=(for_pad(size) - 1)//2, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm2d(nf),
                                        )
        
    def forward(self, x):
        return self.conv_layer(x)


class TripleConv(nn.Module):
    def __init__(self, ni, nf, size=3, stride=1):
        super(TripleConv, self).__init__()
        self.triple_conv = nn.Sequential(ConvBlock(ni, nf),
                                         ConvBlock(nf, ni, size=1),  
                                         ConvBlock(ni, nf)
                                         )
        
    def forward(self, x):
        return self.triple_conv(x)


class DarkCovidNet(nn.Module):
    def __init__(self, output_size=3):
        super(DarkCovidNet, self).__init__()
        self.layers = nn.Sequential(ConvBlock(1, 8),
                                    nn.MaxPool2d(2, stride=2),
                                    ConvBlock(8, 16),
                                    nn.MaxPool2d(2, stride=2),
                                    TripleConv(16, 32),
                                    nn.MaxPool2d(2, stride=2),
                                    TripleConv(32, 64),
                                    nn.MaxPool2d(2, stride=2),
                                    TripleConv(64, 128),
                                    nn.MaxPool2d(2, stride=2),
                                    TripleConv(128, 256),
                                    ConvBlock(256, 128, size=1),
                                    ConvBlock(128, 256),
                                    ConvLayer(256, 2),
                                    nn.Flatten(),
                                    nn.Linear(338, output_size)
                                    )

    def forward(self, x):
        return self.layers(x)
    

class ResNet18(nn.Module):
    def __init__(self, num_classes, is_pretrained=True):
        super().__init__()
        self.resnet = models.resnet18(pretrained=is_pretrained)
        fc_num_features = self.resnet.fc.in_features
        self.classifier = nn.Linear(in_features=fc_num_features,
                                    out_features=num_classes,
                                    bias=True)
        return

    def forward(self, x):
        res = self.resnet.conv1(x)  # bs x 64 x 112 x 112
        res = self.resnet.bn1(res)  # bs x 64 x 112 x 112
        res = self.resnet.relu(res)  # bs x 64 x 112 x 112
        res = self.resnet.maxpool(res)  # bs x 64 x 56 x 56

        l_1 = self.resnet.layer1(res)  # bs x 256 x 56 x 56
        l_2 = self.resnet.layer2(l_1)  # bs x 512 x 28 x 28
        l_3 = self.resnet.layer3(l_2)  # bs x 1024 x 14 x 14
        res = self.resnet.layer4(l_3)  # bs x 1024 x 7 x 7
        g = self.resnet.avgpool(res)  # bs x 2048 x 1 x 1
        out = self.classifier(g.squeeze())  # bs x num_classes
        return out

 
class ResNet50(nn.Module):
    def __init__(self, num_classes, is_pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=is_pretrained)
        fc_num_features = self.resnet.fc.in_features
        self.classifier = nn.Linear(in_features=fc_num_features,
                                    out_features=num_classes,
                                    bias=True)
        return

    def forward(self, x):
        res = self.resnet.conv1(x)  # bs x 64 x 112 x 112
        res = self.resnet.bn1(res)  # bs x 64 x 112 x 112
        res = self.resnet.relu(res)  # bs x 64 x 112 x 112
        res = self.resnet.maxpool(res)  # bs x 64 x 56 x 56

        l_1 = self.resnet.layer1(res)  # bs x 256 x 56 x 56
        l_2 = self.resnet.layer2(l_1)  # bs x 512 x 28 x 28
        l_3 = self.resnet.layer3(l_2)  # bs x 1024 x 14 x 14
        res = self.resnet.layer4(l_3)  # bs x 1024 x 7 x 7
        g = self.resnet.avgpool(res)  # bs x 2048 x 1 x 1
        out = self.classifier(g.squeeze())  # bs x num_classes
        return out
    

class DenseNet(nn.Module):
    def __init__(self, num_classes, is_pretrained=True):
        super().__init__()
        self.densenet = models.densenet121(pretrained=is_pretrained)
        fc_num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features=fc_num_features,
                                    out_features=num_classes,
                                    bias=True)
        return

    def forward(self, x):
        out = self.densenet(x)
        return out
    

class VGG_BN(nn.Module):
    def __init__(self, num_classes, is_pretrained=True):
        super().__init__()
        self.vgg = models.vgg19_bn(pretrained=is_pretrained)
        fc_num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features=fc_num_features,
                                    out_features=num_classes,
                                    bias=True)
        return

    def forward(self, x):
        out = self.vgg(x)
        return out