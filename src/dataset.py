import os
import torch

import numpy as np
import pandas as pd

from PIL import Image, ImageFile
from pathlib import Path

from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class GIDatset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        preprocessing_fn=None,
    ):
        self.image_paths = []
        image_ids = df['id'].values
        self.rle_masks = df['segmentation'].values
        self.preprocessing_fn = preprocessing_fn

        for imgid in image_ids:
            imgid_list = imgid.split('_')
            folder = Path(f"../input/train/{imgid_list[0]}/{imgid_list[0]}_{imgid_list[1]}/scans")
            path = next(folder.glob(f"slice_{imgid_list[3]}_*"), None)
            self.image_paths.append(path)
        
    def __len__(self):
        return len(self.image_paths)
    
    def rle_decode(self, mask_rle, shape):
            if pd.isna(mask_rle) or mask_rle == '':
                return np.zeros(shape, dtype=np.uint8)
            
            s = mask_rle.split()
            starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
            starts -= 1
            ends = starts + lengths
            img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
            for lo, hi in zip(starts, ends):
                img[lo:hi] = 1
            return img.reshape(shape)
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = np.array(img)
        rle_mask = self.rle_masks[index]
        mask = self.rle_decode(rle_mask, shape=img.shape[:2])

        # img = self.preprocessing_fn(img)

        return {
             'image': transforms.ToTensor()(img),
             'mask': transforms.ToTensor()(mask).float()
        }


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    df = pd.read_csv("../input/train.csv")
    import time

    start = time.time()
    dataset = GIDatset(df)
    print(f"Time after storing dataset: {time.time() - start}")
    sample = dataset[334]
    image = sample['image']
    mask = sample['mask']
    print(f"Time after getting image and mask from sample: {time.time() - start}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image.permute(1, 2, 0)) # Assuming (C, H, W) format
    ax[0].set_title("Image")
    ax[1].imshow(mask.squeeze(), cmap='gray') # Assuming (1, H, W)
    ax[1].set_title("Mask")
    plt.show()