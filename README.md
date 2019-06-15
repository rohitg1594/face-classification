# face-classification
Nanonets - Face classification interview problem

## Setup

Run ```./setup.sh``` to setup the data dir. It will download the data from LFW's website. Depending on internet
connection, might take some time.

## Training

Running ```python3 train.py``` with default hyperparmeters will run for 50 epochs and will take ~2 hours with GPU.
It achieves validation accuracy of 75%. Accuracy could potentially be improved with a hyperparameter search. To run
training with different settings, run ```python3 train.py --help``` to see a list of configurable options.

## Architecture

It is a simple architecture based on pre-trained feature extractors.


