from torch.utils.data import Dataset


class FaceDataset(Dataset):
    """Face Classification dataset."""

    def __init__(self, full_dataset, include_subsets, transform=None):
        """
        Args:
            full_dataset (list of dict): Full cross validation dataset.
            include_subsets (list): Subsets to be included in dataloader.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = self._process_dataset(full_dataset, include_subsets)
        self.transform = transform

    @staticmethod
    def _process_dataset(full_dataset, include_subsets):
        ret = []
        subsets = [full_dataset[subset_id] for subset_id in include_subsets]
        for subset in subsets:
            for img1, img2 in subset['match']:
                ret.append([img1, img2, 1])
            for img1, img2 in subset['mismatch']:
                ret.append([img1, img2, 0])

        return ret

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, img2, label = self.dataset[idx]

        sample = {'img1': img1, 'img2': img2, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample
