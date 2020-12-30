import os
from collections import deque
import numpy as np
from PIL import Image, ImageSequence
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def _load_multipage_tiff(path):
    """Load tiff images containing many images in the channel dimension"""
    return [p for p in ImageSequence.Iterator(Image.open(path))]


def _get_val_train_indices(length, fold, ratio=0.8):
    assert 0 < ratio <= 1, "Train/total data ratio must be in range (0.0, 1.0]"
    np.random.seed(0)
    indices = np.arange(0, length, 1, dtype=np.int)
    np.random.shuffle(indices)

    if fold is not None:
        indices = deque(indices)
        indices.rotate(fold * round((1.0 - ratio) * length))
        indices = np.array(indices)
        train_indices = indices[:round(ratio * len(indices))]
        val_indices = indices[round(ratio * len(indices)):]
    else:
        train_indices = indices
        val_indices = []
    return train_indices, val_indices


def data_post_process(img, mask):
    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)
    one = torch.ones_like(mask)
    zero = torch.zeros_like(mask)
    mask = torch.where(mask > 0.5, one, mask)
    mask = torch.where(mask < 0.5, zero, mask)

    return img, mask


def train_data_augmentation(img, mask):
    h_flip = np.random.random()
    v_flip = np.random.random()
    if h_flip > 0.5:
        img = transforms.functional.hflip(img)
        mask = transforms.functional.hflip(mask)
    if v_flip > 0.5:
        img = transforms.functional.vflip(img)
        mask = transforms.functional.vflip(mask)

    left = int(np.random.uniform()*0.3*572)
    right = int((1-np.random.uniform()*0.3)*572)
    top = int(np.random.uniform()*0.3*572)
    bottom = int((1-np.random.uniform()*0.3)*572)
    img = img[:,top:bottom, left:right]
    mask = mask[:,top:bottom, left:right]

    # adjust brightness
    brightness = (torch.rand(img.shape) - 0.5) * (2/5)
    img = img + brightness
    img = img.clip(-1.0, 1.0)

    return img, mask


def create_dataset(data_dir, repeat=400, train_batch_size=16, augment=False, cross_val_ind=1, run_distribute=False):

    images = _load_multipage_tiff(os.path.join(data_dir, 'train-volume.tif'))
    masks = _load_multipage_tiff(os.path.join(data_dir, 'train-labels.tif'))

    train_indices, val_indices = _get_val_train_indices(
        len(images), cross_val_ind)
    train_images = [images[x] for x in train_indices for i in range(repeat)]
    train_masks = [masks[x] for x in train_indices for i in range(repeat)]
    val_images = [images[x] for x in val_indices]
    val_masks = [masks[x] for x in val_indices]

    # transform operators
    t_resize_572 = transforms.Resize(size=(572, 572))
    t_resize_388 = transforms.Resize(size=(388, 388))
    t_pad = transforms.Pad(padding=92)
    t_to_tensor = transforms.ToTensor()
    t_rescale_image = transforms.Normalize(mean=(0.5), std=(0.5))
    t_center_crop = transforms.CenterCrop(size=388)
    trans_image = transforms.Compose([
        t_resize_388,
        t_pad,
        t_to_tensor,
        t_rescale_image
    ])
    trans_mask = transforms.Compose([
        t_resize_388,
        t_pad,
        t_to_tensor
    ])

    # train pre-process
    # pre-process
    train_images = list(map(trans_image, train_images))
    train_masks = list(map(trans_mask, train_masks))
    if run_distribute:
        pass  # TODO
    if augment:
        augment_process = train_data_augmentation
        for i, (img, mask) in tqdm(enumerate(zip(train_images, train_masks))):
            train_images[i], train_masks[i] = augment_process(img, mask)
            train_images[i] = t_resize_572(train_images[i])
            train_masks[i] = t_resize_572(train_masks[i])
    train_masks = list(map(t_center_crop, train_masks))
    # post-process
    for i, (img, mask) in tqdm(enumerate(zip(train_images, train_masks))):
        train_images[i], train_masks[i] = data_post_process(img, mask)
    # to tensor
    train_images = torch.Tensor(np.concatenate(train_images, axis=0))
    train_masks = torch.Tensor(np.concatenate(train_masks, axis=0))
    # make dataset
    train_dataset = TensorDataset(train_images, train_masks)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=True)

    # val pre-process
    val_images = list(map(trans_image, val_images))
    val_masks = list(map(trans_mask, val_masks))
    val_masks = list(map(t_center_crop, val_masks))
    # post-process
    for i, (img, mask) in enumerate(zip(val_images, val_masks)):
        val_images[i], val_masks[i] = data_post_process(img, mask)
    # to tensor
    val_images = torch.Tensor(np.concatenate(val_images, axis=0))
    val_masks = torch.Tensor(np.concatenate(val_masks, axis=0))
    # make dataset
    val_dataset = TensorDataset(val_images, val_masks)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader


def create_val_dataset(data_dir, cross_val_ind=1):

    images = _load_multipage_tiff(os.path.join(data_dir, 'train-volume.tif'))
    masks = _load_multipage_tiff(os.path.join(data_dir, 'train-labels.tif'))

    train_indices, val_indices = _get_val_train_indices(
        len(images), cross_val_ind)
    val_images = [images[x] for x in val_indices]
    val_masks = [masks[x] for x in val_indices]

    # transform operators
    t_resize_388 = transforms.Resize(size=(388, 388))
    t_pad = transforms.Pad(padding=92)
    t_to_tensor = transforms.ToTensor()
    t_rescale_image = transforms.Normalize(mean=(0.5), std=(0.5))
    t_center_crop = transforms.CenterCrop(size=388)
    trans_image = transforms.Compose([
        t_resize_388,
        t_pad,
        t_to_tensor,
        t_rescale_image
    ])
    trans_mask = transforms.Compose([
        t_resize_388,
        t_pad,
        t_to_tensor
    ])

    # val pre-process
    val_images = list(map(trans_image, val_images))
    val_masks = list(map(trans_mask, val_masks))
    val_masks = list(map(t_center_crop, val_masks))
    # post-process
    for i, (img, mask) in enumerate(zip(val_images, val_masks)):
        val_images[i], val_masks[i] = data_post_process(img, mask)
    # to tensor
    val_images = torch.Tensor(np.concatenate(val_images, axis=0))
    val_masks = torch.Tensor(np.concatenate(val_masks, axis=0))
    # make dataset
    val_dataset = TensorDataset(val_images, val_masks)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=True)

    return val_dataloader
