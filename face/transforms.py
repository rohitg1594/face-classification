from skimage import transform
import numpy as np


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img1, img2, label = sample['img1'], sample['img2'], sample['label']

        def transform_img(image):
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)
            img = transform.resize(image, (new_h, new_w))

            return img

        img1 = transform_img(img1)
        img2 = transform_img(img2)

        return {'img1': img1, 'img2': img2, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img1, img2, label = sample['img1'], sample['img2'], sample['label']

        def transform_img(image):
            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[top: top + new_h, left: left + new_w]

            return image

        img1 = transform_img(img1)
        img2 = transform_img(img2)

        return {'img1': img1, 'img2': img2, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img1, img2, label = sample['img1'], sample['img2'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = img1.transpose((2, 0, 1))
        img2 = img2.transpose((2, 0, 1))

        return {'img1': img1, 'img2': img2, 'label': label}
