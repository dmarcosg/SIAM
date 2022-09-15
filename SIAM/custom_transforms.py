import torch
import numpy as np

from skimage import transform

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.transpose(image,(2, 0, 1))
        image = image.astype('float32') / 255

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

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
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        done_resizing = False
        while not done_resizing:
            try:
                img = transform.resize(image, (new_h, new_w),mode='constant',anti_aliasing=True)
                done_resizing = True
            except:
                print('Issue resizing. Trying again.')


        return {'image': img, 'label': label}


class Rotate(object):
    """Rotate the image in a sample to a given size.

    Args:
        output_size (float): Maximum rotation.
    """

    def __init__(self, max_angle):
        assert isinstance(max_angle, (int, float))
        self.max_angle = max_angle

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        angle = (np.random.rand()-0.5)*2*self.max_angle
        done_rotating = False
        while not done_rotating:
            try:
                img = transform.rotate(image, angle)
                done_rotating = True
            except:
                print('Issue rotating. Trying again.')


        return {'image': img, 'label': label}

class VerticalFlip(object):
    """Flip the image in a sample with probability p.

    Args:
        p (float): Probability of vertical flip.
    """

    def __init__(self, p = 0.5):
        assert isinstance(p, (float))
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if np.random.rand() < self.p:
            image = np.fliplr(image)


        return {'image': image, 'label': label}


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
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}