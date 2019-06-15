# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

import pickle
import os
import time
import copy
import argparse

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils import *
from model import PairFaceClassifier
from dataset import FaceDataset
from transforms import *


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, sample in enumerate(dataloaders[phase]):
                print(f"Batch Number: {i}")
                imgs1, imgs2, labels = sample['img1'], sample['img2'], sample['label']
                imgs1 = imgs1.to(device)
                imgs2 = imgs2.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model((imgs1, imgs2))
                    _, preds = torch.max(outputs, 1)
                    print(f"preds: {preds}")
                    loss = criterion(outputs, labels.unsqueeze(dim=1).double())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * imgs1.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    args = parser.parse_args()

    data_path = args.data_path
    img_dir = join(data_path, "lfw-deepfunneled")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cross_dataset_fname = join(data_path, 'cache/cross_dataset.pkl')
    if os.path.exists(join(data_path, 'cache/cross_dataset.pkl')):
        with open(cross_dataset_fname, 'rb') as f:
            cross_dataset = pickle.load(f)
    else:
        cross_dataset = create_dataset(data_path, img_dir)
        with open(cross_dataset_fname, 'wb') as f:
            pickle.dump(cross_dataset, f)

    train_dataset = FaceDataset(cross_dataset,
                                list(range(1)),
                                transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
    valid_dataset = FaceDataset(cross_dataset,
                                [9],
                                transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4),
                   'val': torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=4)}
    dataset_sizes = {'train': len(train_dataset),
                     'val': len(valid_dataset)}
    print(f"Dataset Sizes: {dataset_sizes}")

    model = PairFaceClassifier().double().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)