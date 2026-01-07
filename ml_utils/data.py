from pathlib import Path
import numpy as np
import random
from tifffile import TiffFile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class LandCoverData():
    """Class to represent the S2GLC Land Cover Dataset for the challenge,
    with useful metadata and statistics.
    """
    # image size of the images and label masks
    IMG_SIZE = 256
    # the images are RGB+NIR (4 channels)
    N_CHANNELS = 4
    # we have 9 classes + a 'no_data' class for pixels with no labels (absent in the dataset)
    N_CLASSES = 10
    CLASSES = [
        'no_data',
        'clouds',
        'artificial',
        'cultivated',
        'broadleaf',
        'coniferous',
        'herbaceous',
        'natural',
        'snow',
        'water'
    ]
    # classes to ignore because they are not relevant. "no_data" refers to pixels without
    # a proper class, but it is absent in the dataset; "clouds" class is not relevant, it
    # is not a proper land cover type and images and masks do not exactly match in time.
    IGNORED_CLASSES_IDX = [0, 1]

    # The training dataset contains 18491 images and masks
    # The test dataset contains 5043 images and masks
    TRAINSET_SIZE = 18491
    TESTSET_SIZE = 5043

    # for visualization of the masks: classes indices and RGB colors
    CLASSES_COLORPALETTE = {
        0: [0,0,0],
        1: [255,25,236],
        2: [215,25,28],
        3: [211,154,92],
        4: [33,115,55],
        5: [21,75,35],
        6: [118,209,93],
        7: [130,130,130],
        8: [255,255,255],
        9: [43,61,255]
        }
    CLASSES_COLORPALETTE = {c: np.asarray(color) for (c, color) in CLASSES_COLORPALETTE.items()}

    # statistics
    # the pixel class counts in the training set
    TRAIN_CLASS_COUNTS = np.array(
        [0, 20643, 60971025, 404760981, 277012377, 96473046, 333407133, 9775295, 1071, 29404605]
    )
    # the minimum and maximum value of image pixels in the training set
    TRAIN_PIXELS_MIN = 1
    TRAIN_PIXELS_MAX = 24356

class LandCoverDataset(Dataset):
    """
    Custom PyTorch Dataset for Land Cover images.
    """
    def __init__(self, image_paths, mask_paths=None, transform=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = get_array_from_path(img_path)
        
        mask = None
        if self.mask_paths:
            mask_path = self.mask_paths[idx]
            mask = get_array_from_path(mask_path)
        
        # Apply data augmentation if enabled
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = np.flip(image, axis=1)
                if mask is not None:
                    mask = np.flip(mask, axis=1)
            
            # Random Vertical Flip
            if random.random() > 0.5:
                image = np.flip(image, axis=0)
                if mask is not None:
                    mask = np.flip(mask, axis=0)
            
            # Random Rotation (0, 90, 180, 270 degrees)
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k=k, axes=(0, 1))
                if mask is not None:
                    mask = np.rot90(mask, k=k, axes=(0, 1))
            
            # Ensure contiguous array after flips/rotations for torch compatibility
            image = np.ascontiguousarray(image)
            if mask is not None:
                mask = np.ascontiguousarray(mask)

        # Normalization (Simple min-max scaling approximation)
        # Values are typically 0-2000, max is ~24000. Clipping at 3000 covers most useful info.
        image = np.clip(image / 3000.0, 0, 1)

        # Convert to float32 and normalize if needed (here we keep raw values or normalize)
        # PyTorch expects (C, H, W), but loaded image is (H, W, C)
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1)) # HWC -> CHW
        image = torch.from_numpy(image)

        if mask is not None:
            # Mask is (H, W), PyTorch CrossEntropyLoss expects (H, W) with long type
            mask = mask.astype(np.int64)
            mask = torch.from_numpy(mask)
            return image, mask
        
        return image

def get_array_from_path(path):
    """
    Reads a TIFF file from the given path and returns it as a numpy array.
    """
    with TiffFile(path) as tif:
        return tif.asarray()

def load_dataset(image_paths, mask_paths=None, limit=None):
    """
    Loads images and masks into numpy arrays.
    
    Args:
        image_paths (list): List of paths to images.
        mask_paths (list, optional): List of paths to masks. Defaults to None.
        limit (int, optional): Limit the number of images to load. Defaults to None.
        
    Returns:
        tuple: (X, y) where X is the array of images and y is the array of masks (or None).
    """
    if limit:
        image_paths = image_paths[:limit]
        if mask_paths:
            mask_paths = mask_paths[:limit]

    X = []
    for path in tqdm(image_paths, desc="Loading images"):
        X.append(get_array_from_path(path))
    X = np.array(X)

    y = None
    if mask_paths:
        y = []
        for path in tqdm(mask_paths, desc="Loading masks"):
            y.append(get_array_from_path(path))
        y = np.array(y)
        
    return X, y
