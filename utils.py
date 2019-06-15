from skimage import io, transform
from os.path import join
import random
import matplotlib.pyplot as plt


def get_image(name, img_dir, img_id):
    img_name = '0' * (4 - len(img_id)) + img_id

    return io.imread(join(img_dir, name, '{}_{}.jpg'.format(name, img_name)))


def visualize_dataset(dataset, n=5):
    """
    First two columns plot n random matching faces, and the next two plot random mismatching faces.
    """
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


def parse_pairs(f_name):
    with open(f_name) as f:
        num_sets, num_pairs = [int(p) for p in f.readline().strip().split()]
        subsets = []
        for i in range(num_sets):
            subset = {'match': [f.readline().strip().split() for i in range(num_pairs)],
                      'mismatch': [f.readline().strip().split() for i in range(num_pairs)]}
            subsets.append(subset)

    return subsets


def create_dataset(data_path, img_dir):
    names = parse_pairs(join(data_path, 'pairs.txt'))
    dataset = []

    for i, name_subset in enumerate(names):
        print(f"Loading subset : {i + 1}")
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