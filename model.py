import torch.nn as nn
import torchvision
import torch


class PairFaceClassifier(nn.Module):
    """Classfies if two images belong to same person.

        base_model: CNN model that extracts features of the images.
        hidden_ftrs: Number of hidden ftrs to output from base_model.
    """

    def __init__(self, base_model='resnet18', hidden_ftrs=256):
        super(PairFaceClassifier, self).__init__()
        if base_model == 'resnet18':
            self.model_conv = torchvision.models.resnet18(pretrained=True)
        elif base_model == 'alexnet':
            self.model_conv = torchvision.models.alexnet(pretrained=True)

        num_ftrs = self.model_conv.fc.in_features
        self.model_conv.fc = nn.Linear(num_ftrs, hidden_ftrs)

        self.fc = nn.Linear(2 * hidden_ftrs, 1)

    def forward(self, input):
        img1, img2 = input

        out1 = self.model_conv(img1)
        out2 = self.model_conv(img2)

        score = self.fc(torch.cat((out1, out2), dim=-1)).squeeze(1)

        return score
