import numpy as np
from PIL import ImageFilter, Image
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
import numbers
import random
import math
import warnings
from typing import ClassVar

#import math
#import random
#from PIL import Image
#import numpy as np
import torch
from torchvision.transforms import Normalize

class Denormalize(Normalize):
    

    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        super().__init__((-mean / std).tolist(), (1 / std).tolist())


def wrapper(transform: ClassVar):
    
    class WrapperTransform(transform):
        def __call__(self, image, **kwargs):
            image = super().__call__(image)
            return image, kwargs
    return WrapperTransform


ToTensor = wrapper(T.ToTensor)
Normalize = wrapper(T.Normalize)
ColorJitter = wrapper(T.ColorJitter)


def resize(image: Image.Image, size: int, interpolation=Image.BILINEAR,
           keypoint2d: np.ndarray=None, intrinsic_matrix: np.ndarray=None):
    width, height = image.size
    assert width == height
    factor = float(size) / float(width)
    image = F.resize(image, size, interpolation)
    keypoint2d = np.copy(keypoint2d)
    keypoint2d *= factor
    intrinsic_matrix = np.copy(intrinsic_matrix)
    intrinsic_matrix[0][0] *= factor
    intrinsic_matrix[0][2] *= factor
    intrinsic_matrix[1][1] *= factor
    intrinsic_matrix[1][2] *= factor
    return image, keypoint2d, intrinsic_matrix


def crop(image: Image.Image, top, left, height, width, keypoint2d: np.ndarray, prf = False):
    #if prf: print("transforms-crop: input",(top, left, height, width)) #(36, 85, 159, 159)
    #if prf: print("transforms-crop: original image", image.size) #(320, 320)
    #if prf: print("transforms-crop: original image", image)
    image = F.crop(image, top, left, height, width)
    #if prf: print("transforms-crop: crop image", image.size) #(159, 159)
    #if prf: print("transforms-crop: crop image", image)
    keypoint2d = np.copy(keypoint2d)
    #if prf: print("transforms-crop: original keypoints2d", keypoint2d)
    keypoint2d[:, 0] -= left # -85 left as zero for coordinates
    keypoint2d[:, 1] -= top # -36 top as zero for coordinates
    #if prf: print("transforms-crop: crop keypoints2d", keypoint2d)
    return image, keypoint2d


def resized_crop(img, top, left, height, width, size, interpolation=Image.BILINEAR,
                 keypoint2d: np.ndarray=None, intrinsic_matrix: np.ndarray=None):
   
    assert isinstance(img, Image.Image), 'img should be PIL Image'
    img, keypoint2d = crop(img, top, left, height, width, keypoint2d)
    img, keypoint2d, intrinsic_matrix = resize(img, size, interpolation, keypoint2d, intrinsic_matrix)
    return img, keypoint2d, intrinsic_matrix


def center_crop(image, output_size, keypoint2d: np.ndarray):
    
    width, height = image.size
    crop_height, crop_width = output_size
    crop_top = int(round((height - crop_height) / 2.))
    crop_left = int(round((width - crop_width) / 2.))
    return crop(image, crop_top, crop_left, crop_height, crop_width, keypoint2d)


def hflip(image: Image.Image, keypoint2d: np.ndarray):
    width, height = image.size
    image = F.hflip(image)
    keypoint2d = np.copy(keypoint2d)
    keypoint2d[:, 0] = width - 1. - keypoint2d[:, 0]
    return image, keypoint2d


def rotate(image: Image.Image, angle, keypoint2d: np.ndarray):
    image = F.rotate(image, angle)

    angle = -np.deg2rad(angle)
    keypoint2d = np.copy(keypoint2d)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    width, height = image.size
    keypoint2d[:, 0] = keypoint2d[:, 0] - width / 2
    keypoint2d[:, 1] = keypoint2d[:, 1] - height / 2
    keypoint2d = np.matmul(rotation_matrix, keypoint2d.T).T
    keypoint2d[:, 0] = keypoint2d[:, 0] + width / 2
    keypoint2d[:, 1] = keypoint2d[:, 1] + height / 2
    return image, keypoint2d


def resize_pad(img, keypoint2d, size, interpolation=Image.BILINEAR):
    w, h = img.size
    if w < h:
        oh = size
        ow = int(size * w / h)
        img = img.resize((ow, oh), interpolation)
        pad_top = pad_bottom = 0
        pad_left = math.floor((size - ow) / 2)
        pad_right = math.ceil((size - ow) / 2)
        keypoint2d = keypoint2d * oh / h
        keypoint2d[:, 0] += (size - ow) / 2
    else:
        ow = size
        oh = int(size * h / w)
        img = img.resize((ow, oh), interpolation)
        pad_top = math.floor((size - oh) / 2)
        pad_bottom = math.ceil((size - oh) / 2)
        pad_left = pad_right = 0
        keypoint2d = keypoint2d * ow / w
        keypoint2d[:, 1] += (size - oh) / 2
        keypoint2d[:, 0] += (size - ow) / 2
    img = np.asarray(img)

    img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=0)
    return Image.fromarray(img), keypoint2d


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, **kwargs):
        for t in self.transforms:
            image, kwargs = t(image, **kwargs)
        return image, kwargs


class GaussianBlur(object):
    def __init__(self, low=0, high=0.8):
        self.low = low
        self.high = high

    def __call__(self, image: Image, **kwargs):
        radius = np.random.uniform(low=self.low, high=self.high)
        image = image.filter(ImageFilter.GaussianBlur(radius))
        return image, kwargs


class Resize(object):
    """Resize the input PIL Image to the given size.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, keypoint2d: np.ndarray, intrinsic_matrix: np.ndarray, **kwargs):
        image, keypoint2d, intrinsic_matrix = resize(image, self.size, self.interpolation, keypoint2d, intrinsic_matrix)
        kwargs.update(keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        if 'depth' in kwargs:
            kwargs['depth'] = F.resize(kwargs['depth'], self.size)
        return image, kwargs


class ResizePad(object):
    """Pad the given image on all sides with the given "pad" value to resize the image to the given size.
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, keypoint2d, **kwargs):
        image, keypoint2d = resize_pad(img, keypoint2d, self.size, self.interpolation)
        kwargs.update(keypoint2d=keypoint2d)
        return image, kwargs


class CenterCrop(object):
    """Crops the given PIL Image at the center.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, image, keypoint2d, **kwargs):
       
        image, keypoint2d = center_crop(image, self.size, keypoint2d)
        kwargs.update(keypoint2d=keypoint2d)
        if 'depth' in kwargs:
            kwargs['depth'] = F.center_crop(kwargs['depth'], self.size)
        return image, kwargs


class RandomRotation(object):
   
    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees


    @staticmethod
    def get_params(degrees):
        
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, image, keypoint2d, **kwargs):
        

        angle = self.get_params(self.degrees)

        image, keypoint2d = rotate(image, angle, keypoint2d)
        kwargs.update(keypoint2d=keypoint2d)
        if 'depth' in kwargs:
            kwargs['depth'] = F.rotate(kwargs['depth'], angle)
        return image, kwargs


class RandomResizedCrop(object):
    

    def __init__(self, size, scale=(0.6, 1.3), interpolation=Image.BILINEAR):
        self.size = size
        if scale[0] > scale[1]:
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale

    @staticmethod
    def get_params(img, scale):
        
        width, height = img.size
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = 1

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to whole image
        return 0, 0, height, width

    def __call__(self, image, keypoint2d: np.ndarray, intrinsic_matrix: np.ndarray, **kwargs):
        
        i, j, h, w = self.get_params(image, self.scale)
        image, keypoint2d, intrinsic_matrix = resized_crop(image, i, j, h, w, self.size, self.interpolation, keypoint2d, intrinsic_matrix)
        kwargs.update(keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        if 'depth' in kwargs:
            kwargs['depth'] = F.resized_crop(kwargs['depth'], i, j, h, w, self.size, self.interpolation,)
        return image, kwargs


class RandomApply(T.RandomTransforms):
    

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, image, **kwargs):
        if self.p < random.random():
            return image, kwargs
        for t in self.transforms:
            image, kwargs = t(image, **kwargs)
        return image, kwargs
