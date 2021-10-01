import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

import torch
from torch import nn
from torchvision.transforms import transforms

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)





def get_cifar10(args, root, active_index = None):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, labeled_idxs_org = x_u_split(
        args, base_dataset.targets, active_index)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    train_labeled_sim_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=ContrastiveLearningViewGenerator(32,args.n_views))
    
    
    train_unlabeled_sim_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=ContrastiveLearningViewGenerator(32,args.n_views))


    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, labeled_idxs_org, train_unlabeled_sim_dataset, train_labeled_sim_dataset


def get_cifar100(args, root, active_index = None):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, labeled_idxs_org = x_u_split(
        args, base_dataset.targets, active_index)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    train_labeled_sim_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=ContrastiveLearningViewGenerator(32,args.n_views))
    
    
    train_unlabeled_sim_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=ContrastiveLearningViewGenerator(32,args.n_views))    
        

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, labeled_idxs_org, train_unlabeled_sim_dataset, train_labeled_sim_dataset


def x_u_split(args, labels, act_index = None):
    if act_index is None:
        labels = np.array(labels)
        labeled_idx = []
        # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
        unlabeled_idx = np.array(range(len(labels)))
            
        idx = np.arange(len(labels))
        idx = np.random.choice(idx, args.num_labeled, False)
        labeled_idx.extend(idx)

        labeled_idx = np.array(labeled_idx)
        assert len(labeled_idx) == args.num_labeled
        labeled_idx_org = labeled_idx
        labeled_num = args.num_labeled
        if args.expand_labels or args.num_labeled < args.batch_size:
            num_expand_x = math.ceil(
                args.batch_size * args.eval_step / labeled_num) 
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        
    else:
        labeled_idx = act_index
        unlabeled_idx = np.array(range(len(labels)))
        labeled_idx_org = labeled_idx
        labeled_num = act_index.shape[0]

        if args.expand_labels or args.num_labeled < args.batch_size:
            num_expand_x = math.ceil(
                args.batch_size * args.num_sample / labeled_num) 
                
            if act_index is not None and act_index.shape[0] == args.stop_active:
                num_expand_x = math.ceil(
                args.batch_size * args.eval_step / args.stop_active)

        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    return labeled_idx, unlabeled_idx, labeled_idx_org


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
        

#def get_simclr_pipeline_transform(size, s=1):
#    """Return a set of data augmentation transformations as described in the SimCLR paper."""
#    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
#                                          transforms.RandomHorizontalFlip(),
#                                          transforms.RandomApply([color_jitter], p=0.8),
#                                          transforms.RandomGrayscale(p=0.2),
#                                          GaussianBlur(kernel_size=int(0.1 * size)),
#                                          transforms.ToTensor()])
        
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, size, n_views=2):
        self.color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1) #s = 1; size = 32
        self.base_transform = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([self.color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * size)),
                                          transforms.ToTensor()])
        self.n_views = n_views

    def __call__(self, x):
        #return [self.base_transform(x) for i in range(self.n_views)]
        return self.base_transform(x), self.base_transform(x)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img



DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}
