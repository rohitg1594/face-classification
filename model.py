import torch.nn as nn
import torchvision
import torch

from utils import set_parameter_requires_grad

class PairFaceClassifier(nn.Module):
    """Classfies if two images belong to same person.

        base_model: CNN model that extracts features of the images.
        hidden_ftrs: Number of hidden ftrs to output from base_model.
    """

    def __init__(self, base_model='resnet18', hidden_ftrs=256, feature_extract=False):
        super(PairFaceClassifier, self).__init__()
        if base_model == 'resnet18':
            self.model_conv = torchvision.models.resnet18(pretrained=True)
            set_parameter_requires_grad(self.model_conv, feature_extract)
            num_ftrs = self.model_conv.fc.in_features
            self.model_conv.fc = nn.Linear(num_ftrs, hidden_ftrs)
        elif base_model == 'alexnet':
            self.model_conv = torchvision.models.alexnet(pretrained=True)
            set_parameter_requires_grad(self.model_conv, feature_extract)
            num_ftrs = self.model_conv.classifier[6].in_features
            self.model_conv.classifier[6] = nn.Linear(num_ftrs, hidden_ftrs)
        elif base_model == 'vgg16':
            self.model_conv = torchvision.models.vgg11(pretrained=True)
            set_parameter_requires_grad(self.model_conv, feature_extract)
            num_ftrs = self.model_conv.classifier[6].in_features
            self.model_conv.classifier[6] = nn.Linear(num_ftrs, hidden_ftrs)
        elif base_model == 'squeezenet':
            self.model_conv = torchvision.models.squeezenet1_0(pretrained=True)
            set_parameter_requires_grad(self.model_conv, feature_extract)
            self.model_conv.classifier[1] = nn.Conv2d(512, hidden_ftrs, kernel_size=(1, 1), stride=(1, 1))
            self.model_conv.num_classes = hidden_ftrs
        else:
            raise RuntimeError("{} not supported.".format(base_model))

        self.fc = nn.Linear(2 * hidden_ftrs, 1)

    def forward(self, input):
        img1, img2 = input

        out1 = self.model_conv(img1)
        out2 = self.model_conv(img2)

        score = self.fc(torch.cat((out1, out2), dim=-1)).squeeze(1)

        return score
