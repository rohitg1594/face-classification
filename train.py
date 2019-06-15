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
from torchvision import transforms

from utils import *
from model import PairFaceClassifier
from dataset import FaceDataset
from transforms import *


def train_model(args, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, sample in enumerate(dataloaders[phase]):
                print("Batch Number: {}".format(i))
                imgs1, imgs2, labels = sample['img1'], sample['img2'], sample['label']
                imgs1 = imgs1.to(device)
                imgs2 = imgs2.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model((imgs1, imgs2))
                    preds = torch.gt(outputs, 0).double()
                    loss = criterion(outputs, labels.double())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * imgs1.size(0)
                correct = torch.sum(preds == labels.data.double())
                running_corrects += correct
                print("Running Acc: {}".format(running_corrects.double() / (args.batch_size * (i + 1))))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--base-model", type=str,
                        choices=['alexnet', 'squeezenet', 'resnet18', 'vgg16'],
                        default='resnet18')
    parser.add_argument("--hidden-ftrs", type=int, default=256)
    parser.add_argument("--rescale", type=int, default=256)
    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--device", type=str, default=0)
    parser.add_argument("--train-sets", type=eval, default=list(range(9)))
    parser.add_argument("--valid-sets", type=eval, default=[9])
    args = parser.parse_args()

    return args


def get_dataset(data_path):
    cross_dataset_fname = join(data_path, 'cache/cross_dataset.pkl')
    if os.path.exists(join(data_path, 'cache/cross_dataset.pkl')):
        with open(cross_dataset_fname, 'rb') as f:
            cross_dataset = pickle.load(f)
    else:
        cross_dataset = create_dataset(data_path, img_dir)
        if not os.path.exists(join(data_path, 'cache')):
            os.makedirs(join(data_path, 'cache'))
        with open(cross_dataset_fname, 'wb') as f:
            pickle.dump(cross_dataset, f)

    return cross_dataset


if __name__ == "__main__":

    args = get_args()
    data_path = args.data_path
    img_dir = join(data_path, "lfw-deepfunneled")
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print("Using device: {}".format(device))

    cross_dataset = get_dataset(data_path)

    train_dataset = FaceDataset(cross_dataset,
                                args.train_sets,
                                transform=transforms.Compose([Rescale(args.rescale), RandomCrop(args.crop), ToTensor()]))
    valid_dataset = FaceDataset(cross_dataset,
                                args.valid_sets,
                                transform=transforms.Compose([Rescale(args.rescale), RandomCrop(args.crop), ToTensor()]))
    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers),
                   'val': torch.utils.data.DataLoader(valid_dataset,
                                                      batch_size=args.batch_size,
                                                      shuffle=False,
                                                      num_workers=args.num_workers)}
    dataset_sizes = {'train': len(train_dataset),
                     'val': len(valid_dataset)}
    print("Dataset Sizes: {}".format(dataset_sizes))

    model = PairFaceClassifier(base_model=args.base_model, hidden_ftrs=args.hidden_ftrs).double().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(args, model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.num_epochs)
