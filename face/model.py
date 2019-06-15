import torch.nn as nn
import torchvision
import torch

from face.utils import set_parameter_requires_grad


class PairFaceClassifier(nn.Module):
    """Classfies if two images belong to same person.

        base_model: CNN model that extracts features of the images.
        hidden_ftrs: Number of hidden ftrs to output from base_model.
    """

    def __init__(self, base_model='resnet18', feature_extract=False, dropout=0.3):
        super(PairFaceClassifier, self).__init__()
        if base_model == 'resnet18':
            self.model_conv = torchvision.models.resnet18(pretrained=True)
            set_parameter_requires_grad(self.model_conv, feature_extract)
            num_ftrs = self.model_conv.fc.in_features
            self.model_conv.fc = nn.Sequential()  # Simulate identity layer
        elif base_model == 'alexnet':
            self.model_conv = torchvision.models.alexnet(pretrained=True)
            set_parameter_requires_grad(self.model_conv, feature_extract)
            num_ftrs = self.model_conv.classifier[6].in_features
            self.model_conv.classifier[6] = nn.Sequential()
        elif base_model == 'vgg16':
            self.model_conv = torchvision.models.vgg11(pretrained=True)
            set_parameter_requires_grad(self.model_conv, feature_extract)
            num_ftrs = self.model_conv.classifier[6].in_features
            self.model_conv.classifier[6] = nn.Sequential()
        else:
            raise RuntimeError("{} not supported.".format(base_model))

        self.fc1 = nn.Linear(2 * num_ftrs, num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, 1)
        self.dp = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.classifier = nn.Sequential(self.dp,
                                        self.fc1,
                                        self.relu,
                                        self.dp,
                                        self.fc2)

    def forward(self, input):
        img1, img2 = input

        out1 = self.model_conv(img1)
        out2 = self.model_conv(img2)

        score = self.classifier(torch.cat((out1, out2), dim=-1))

        return score.squeeze(1)
