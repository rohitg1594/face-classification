from skimage import io
from os.path import join
import os
import random
import matplotlib.pyplot as plt
import pickle


# Load image from disk
def get_image(name, img_dir, img_id):
    img_name = '0' * (4 - len(img_id)) + img_id

    return io.imread(join(img_dir, name, '{}_{}.jpg'.format(name, img_name)))


# First two columns plot n random matching faces, and the next two plot random mismatching faces.
def visualize_dataset(dataset, n=5):

    subset_id = random.randint(0, 9)
    subset = dataset[subset_id]

    match_pair_ids = random.sample(list(range(len(subset['match']))), n)
    mismatch_pair_ids = random.sample(list(range(len(subset['mismatch']))), n)

    plt.figure(figsize=(20, 10))

    plt.subplot(n, 1, 1)
    for i, pair_id in enumerate(match_pair_ids):
        plt.subplot(n, 4, i * 4 + 1)
        plt.imshow(subset['match'][pair_id][0])

        plt.subplot(n, 4, i * 4 + 2)
        plt.imshow(subset['match'][pair_id][1])

    for i, pair_id in enumerate(mismatch_pair_ids):
        plt.subplot(n, 4, i * 4 + 3)
        plt.imshow(subset['mismatch'][pair_id][0])

        plt.subplot(n, 4, i * 4 + 4)
        plt.imshow(subset['mismatch'][pair_id][1])


# Plots a single sample that comes from dataloader
def plot_sample(sample):
    img1, img2 = sample['img1'], sample['img2']

    plt.subplot(1, 2, 1)
    plt.imshow(img1.transpose((1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.imshow(img2.transpose((1, 2, 0)))

    label = sample['label']
    if label:
        plt.suptitle('MATCH')
    else:
        plt.suptitle('MISMATCH')


# Parses pairs.txt file into cross validation format
def parse_pairs(f_name):
    with open(f_name) as f:
        num_sets, num_pairs = [int(p) for p in f.readline().strip().split()]
        subsets = []
        for i in range(num_sets):
            subset = {'match': [f.readline().strip().split() for _ in range(num_pairs)],
                      'mismatch': [f.readline().strip().split() for _ in range(num_pairs)]}
            subsets.append(subset)

    return subsets


# Creates a dataset split into subsets, that can be used for cross validation
def create_dataset(data_path, img_dir):
    names = parse_pairs(join(data_path, 'pairs.txt'))
    dataset = []

    for i, name_subset in enumerate(names):
        print("Loading subset : {}".format(i * 1))
        img_subset = {'match': [],
                      'mismatch': []}

        match_names = name_subset['match']
        for pair in match_names:
            name, img_id1, img_id2 = pair
            img1 = get_image(name, img_dir, img_id1)
            img2 = get_image(name, img_dir, img_id2)
            img_subset['match'].append([img1, img2])

        mismatch_names = name_subset['mismatch']
        for pair in mismatch_names:
            name1, img_id1, name2, img_id2 = pair
            img1 = get_image(name1, img_dir, img_id1)
            img2 = get_image(name2, img_dir, img_id2)
            img_subset['mismatch'].append([img1, img2])

        dataset.append(img_subset)

    return dataset


# Looks for dataset in cache, otherwise computes it
def get_dataset(data_path, img_dir):
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


# If using base model as feature extractor, then don't compute gradients
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
