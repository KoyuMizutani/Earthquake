import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from math import exp

class BaselineDataset(Dataset):
    def __init__(self, root=None, mode="train"):
        self.root = root
        # 全てのデータのパスを入れる
        data_dir = os.path.join(self.root, mode)
        self.all_data = glob.glob(data_dir + "/*")
        # all_dataは一次元配列

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        with open(self.all_data[idx], "r") as f:
            txt = f.readlines()[0]
        x, y, depth, mag = txt.split(",")
        x, y, depth, mag = int(x), int(y), float(depth), float(mag)
        lbl_data = np.loadtxt(self.all_data[idx], delimiter=',', dtype="float32", skiprows=1)
        return x, y, depth, mag, lbl_data


class ClsDataset(Dataset):
    def __init__(self, root=None, mode="train", transform=None, input_width=15, input_dim=1):
        assert input_width % 2 == 1 # 奇数であることを確かめる
        self.root = root
        self.transform = transform
        self.input_width = input_width
        self.input_dim = input_dim + 1 # depthを含める

        # 全てのデータのパスを入れる
        data_dir = os.path.join(self.root, mode)
        self.all_data = glob.glob(data_dir + "/*")
        # all_dataは一次元配列

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        with open(self.all_data[idx], "r") as f:
            txt = f.readlines()[0]
        x, y, depth, mag = txt.split(",")
        x, y, depth, mag = int(x), int(y), float(depth), float(mag)
        lbl_data = np.loadtxt(self.all_data[idx], delimiter=',', dtype=int, skiprows=1)
        len_data = len(lbl_data)
        img = torch.zeros(self.input_dim, len(lbl_data), len(lbl_data))
        half = self.input_width//2
        for i in range(x - half, x + half + 1):
            for j in range(y - half, y + half + 1):
                if 0 <= i < len_data and 0 <= j < len_data:
                    img[0][i][j] = depth
                    for k in range(1, self.input_dim):
                        img[k][i][j] = exp(mag)
        return img, lbl_data

class RegDataset(Dataset):
    def __init__(self, root=None, mode="train", input_width=15, input_dim=1):
        assert input_width % 2 == 1 # 奇数であることを確かめる
        self.root = root
        self.input_width = input_width
        self.input_dim = input_dim + 1 # depthを含める

        # 全てのデータのパスを入れる
        data_dir = os.path.join(self.root, mode)
        self.all_data = glob.glob(data_dir + "/*")
        # all_dataは一次元配列

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        with open(self.all_data[idx], "r") as f:
            txt = f.readlines()[0]
        x, y, depth, mag = txt.split(",")
        x, y, depth, mag = int(x), int(y), float(depth), float(mag)
        lbl_data = np.loadtxt(self.all_data[idx], delimiter=',', dtype="float32", skiprows=1)
        len_data = len(lbl_data)
        img = torch.zeros(self.input_dim, len(lbl_data), len(lbl_data))
        half = self.input_width//2
        for i in range(x - half, x + half + 1):
            for j in range(y - half, y + half + 1):
                if 0 <= i < len_data and 0 <= j < len_data:
                    img[0][i][j] = depth/1000
                    for k in range(1, self.input_dim):
                        img[k][i][j] = (mag/10)**k
        return img, lbl_data