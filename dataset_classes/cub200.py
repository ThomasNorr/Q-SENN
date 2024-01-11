# Dataset should lie under /root/
# root is currently set to ~/tmp/Datasets/CUB200
# If cropped iamges, like for PIP-Net, ProtoPool, etc. are used, then the crop_root should be set to a folder containing the
# cropped images in the expected structure, obtained by following ProtoTree's instructions.
# https://github.com/M-Nauta/ProtoTree/blob/main/README.md#preprocessing-cub
import os
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from dataset_classes.utils import txt_load


class CUB200Class(Dataset):
    root = Path.home() / "tmp/Datasets/CUB200"
    crop_root = Path.home() / "tmp/Datasets/PPCUB200"
    base_folder = 'CUB_200_2011/images'
    def __init__(self,  train, transform, crop=True):
        self.train = train
        self.transform = transform
        self.crop = crop
        self._load_metadata()
        self.loader = default_loader

        if crop:
            self.adapt_to_crop()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def adapt_to_crop(self):
      #  ds_name = [x for x in self.cropped_dict.keys() if x in self.root][0]
        self.root = self.crop_root
        folder_name = "train" if self.train else "test"
        folder_name = folder_name + "_cropped"
        self.base_folder = 'CUB_200_2011/' + folder_name

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        img = self.transform(img)
        return img, target

    @classmethod
    def get_image_attribute_labels(self, train=False):
        image_attribute_labels = pd.read_csv(
            os.path.join('/home/norrenbr/tmp/Datasets/CUB200', 'CUB_200_2011', "attributes",
                         'image_attribute_labels.txt'),
            sep=' ', names=['img_id', 'attribute', "is_present", "certainty", "time"], on_bad_lines="skip")
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        merged = image_attribute_labels.merge(train_test_split, on="img_id")
        filtered_data = merged[merged["is_training_img"] == train]
        return filtered_data


    @staticmethod
    def filter_attribute_labels(labels, min_certainty=3):
        is_invisible_present = labels[labels["certainty"] == 1]["is_present"].sum()
        if is_invisible_present != 0:
            raise ValueError("Invisible present")
        labels["img_id"] -= min(labels["img_id"])
        labels["img_id"] = fillholes_in_array(labels["img_id"])
        labels[labels["certainty"] == 1]["certainty"] = 4
        labels = labels[labels["certainty"] >= min_certainty]
        labels["attribute"] -= min(labels["attribute"])
        labels = labels[["img_id", "attribute", "is_present"]]
        labels["is_present"] = labels["is_present"].astype(bool)
        return labels



def fillholes_in_array(array):
    unique_values = np.unique(array)
    mapping = {x: i for i, x in enumerate(unique_values)}
    array = array.map(mapping)
    return array
