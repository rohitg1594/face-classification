# face-classification
Nanonets - Face classification interview problem.

## Setup
Run ```./setup.sh``` to setup the data dir. It will download the data from LFW's website and install all the 
dependencies. Depending on internet connection, it might take some time.

## Dataset
The dataset is constructed based on the pairs.txt file on LFW website. It is split into 10 subsets of 600 pairs each.
Of these 300 are matching and 300 do not match. We take the first 9 subets for training and the last subset for validation.
This results in a train set of size 5400 pairs and a valid set of size 600 pairs. Valid and train sets can be changed using 
```--valid-sets``` and ```--test-sets``` options in ```train.py```.
 

## Training

Running ```python3 train.py``` with default hyperparmeters will run for 50 epochs and will take ~2 hours with GPU.
It achieves validation accuracy of 75%. Accuracy could potentially be improved with a hyperparameter search. To run
training with different settings, run ```python3 train.py --help``` to see a list of configurable options.

## Architecture

It is a simple architecture based on pre-trained feature extractors (default is resnet18, can change using 
```--base-model``` option). The two images in a pair are sent to base model to extract features. The features are 
concatenated and passed through a MLP with one hidden layer with relu activation and dropout.
