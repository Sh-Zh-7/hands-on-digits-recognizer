import os
from PIL import Image
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch.utils.data as Data
import torchvision.transforms as transform

# Define transforms
train_transforms = transform.Compose([
    transform.RandomCrop((25, 25)),
    transform.RandomRotation(10),
    transform.Resize((28, 28)),
    transform.ToTensor(),
    transform.Normalize((0.5,), (0.5,))
])
eval_transforms = transform.Compose([
    transform.ToTensor(),
    transform.Normalize((0.5,), (0.5,))
])

# Helper
def GetDataSet(filename, data_type, test_size=0.2, random_seed=0):
    """ Get x and y by passing type of data and data path """
    data_set = pd.read_csv(filename)
    if data_type == "test":
        return data_set.values
    else:
        labels = data_set["label"]
        data_set = data_set.drop("label", axis=1)
        train_x, val_x, train_y, val_y = train_test_split(
            data_set, labels, test_size=test_size, random_state=random_seed
        )
        if data_type == "train":
            return train_x.values, train_y.values
        elif data_type == "val":
            return val_x.values, val_y.values
        else:
            raise RuntimeError("Unknown parameters!")


class DigitsDataSet(Data.Dataset):
    def __init__(self, filename, data_type, transform):
        super(DigitsDataSet, self).__init__()
        self.x, self.y = GetDataSet(filename, data_type)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # Only accept float value
        img_x = Image.fromarray(np.array(self.x[index].reshape(28, 28), dtype="float"))      # Convert into image object
        ret_x = self.transform(img_x)
        ret_y = self.y[index]
        return ret_x, ret_y

def GetDataLoader(dirname, data_types, params):
    """
    Get data loader by passing dirname and dataset types,
    Note we can get train dataset and cv dataset at the same time
    """
    data_loaders = {}
    types = ["train", "val", "test"]
    for split in types:
        if split in data_types:
            file_path = os.path.join(dirname, "test.csv") if split == "test" \
                        else os.path.join(dirname, "train.csv")
            if split == "train":
                dl = Data.DataLoader(
                    dataset=DigitsDataSet(file_path, split, train_transforms),
                    batch_size=params.batch_size,
                    shuffle=True,
                    num_workers=params.num_workers
                )
            else:
                dl = Data.DataLoader(
                    dataset=DigitsDataSet(file_path, split, eval_transforms),
                    batch_size=params.batch_size,
                    shuffle=False,
                    num_workers=params.num_workers
                )
            data_loaders[split] = dl
    return data_loaders
