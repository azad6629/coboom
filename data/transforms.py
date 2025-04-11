import cv2
import numpy as np
from PIL import ImageFile
from PIL import Image, ImageOps
import torch
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GaussianBlur():
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))

class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)

def get_transform(image_size, gb_prob=1.0, solarize_prob=0.):
    t_list = []
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    t_list = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=23)], p=gb_prob),
            transforms.RandomApply([Solarize()], p=solarize_prob),
            transforms.ToTensor(),
            normalize]
    
    transform = transforms.Compose(t_list)
    return transform


def get_vae_transform():
    mean = [0.4760, 0.4760, 0.4760]
    std  = [0.3001, 0.3001, 0.3001]
    imgtransCrop = 224

    # Tranform data
    normalize = transforms.Normalize(mean, std)
    transformList = []
    transformList.append(transforms.Resize((imgtransCrop, imgtransCrop)))
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    transform = transforms.Compose(transformList)
    
    return transform
