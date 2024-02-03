import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import math
from geopy.distance import geodesic

MASK = np.loadtxt("/home/mizutani/exp/EQPrediction/AI/src/mask.csv", delimiter=',', dtype=int)

INPUT_WIDTH = 64
START_LONGITUDE = 128.0
START_LATITUDE = 46.0
END_LONGITUDE = 146.0
END_LATITUDE = 30.0

class BaselineNetwork(nn.Module):
    def __init__(self, input_width=INPUT_WIDTH, start_longitude=START_LONGITUDE, end_longitude=END_LONGITUDE, start_latitude=START_LATITUDE, end_latitude=END_LATITUDE, mask=MASK):
        super(BaselineNetwork, self).__init__()
        self.input_width = input_width
        self.start_longitude = torch.tensor(start_longitude, dtype=torch.float32).mul_(math.pi/180)  # Convert degree to radian
        self.end_longitude = torch.tensor(end_longitude, dtype=torch.float32).mul_(math.pi/180)  # Convert degree to radian
        self.start_latitude = torch.tensor(start_latitude, dtype=torch.float32).mul_(math.pi/180)  # Convert degree to radian
        self.end_latitude = torch.tensor(end_latitude, dtype=torch.float32).mul_(math.pi/180)  # Convert degree to radian
        self.mask = torch.tensor(mask, dtype=torch.float32)  # Convert numpy array to torch tensor 
        # Initialize amp as a learnable parameter
        self.amp = nn.Parameter(torch.ones(self.input_width, self.input_width))


    def forward(self, x, y, depth, mag):
        batch_size = len(x)

        # Calculate grid of coordinates for cells
        cell_width = (self.end_longitude - self.start_longitude) / self.input_width
        cell_longs = torch.linspace(self.start_longitude + cell_width / 2, self.end_longitude - cell_width / 2, self.input_width).unsqueeze(0).expand(self.input_width, -1).to(x.device)  # shape: [input_width, input_width]
        cell_height = (self.end_latitude - self.start_latitude) / self.input_width
        cell_lats = torch.linspace(self.start_latitude + cell_height / 2, self.end_latitude - cell_height / 2, self.input_width).unsqueeze(-1).expand(-1, self.input_width).to(x.device)  # shape: [input_width, input_width]

        # Calculate epicenter coordinates for each batch
        epicenter_longs = self.start_longitude + (self.end_longitude - self.start_longitude) * (x.view(batch_size, 1, 1) + 0.5) / self.input_width  # shape: [batch_size, 1, 1]
        epicenter_lats = self.start_latitude + (self.end_latitude - self.start_latitude) * (y.view(batch_size, 1, 1) + 0.5) / self.input_width  # shape: [batch_size, 1, 1]

        # Haversine formula to calculate the distance between two points on the earth
        dlon = cell_longs - epicenter_longs  # shape: [batch_size, input_width, input_width]
        dlat = cell_lats - epicenter_lats  # shape: [batch_size, input_width, input_width]
        a = torch.sin(dlat / 2)**2 + torch.cos(epicenter_lats) * torch.cos(cell_lats) * torch.sin(dlon / 2)**2  # shape: [batch_size, input_width, input_width]
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))  # shape: [batch_size, input_width, input_width]
        X = torch.sqrt((6371.0 * c) ** 2 + depth.view(batch_size, 1, 1) ** 2)  # shape: [batch_size, input_width, input_width]

        log_PGV_b = 0.58 * mag.view(batch_size, 1, 1) + 0.0038 * depth.view(batch_size, 1, 1) - 1.29 - torch.log(X + 0.0028 * 10 ** (0.5 * mag.view(batch_size, 1, 1))) - 0.002 * X  # shape: [batch_size, input_width, input_width]
        PGV_b = torch.exp(log_PGV_b)  # shape: [batch_size, input_width, input_width]
        PGV = PGV_b * self.amp.to(x.device)  # shape: [batch_size, input_width, input_width]
        I_high = 2.002 + 2.603 * torch.log(PGV) - 0.213 * (torch.log(PGV)) ** 2  # shape: [batch_size, input_width, input_width]
        I_low = 2.165 + 2.262 * torch.log(PGV)  # shape: [batch_size, input_width, input_width]
        h = torch.where(I_high > 4.0, I_high, I_low)  # shape: [batch_size, input_width, input_width]

        # Apply mask
        h = h * self.mask.to(x.device)  # shape: [batch_size, input_width, input_width]

        return h


class ClsNetwork(nn.Module):
    def __init__(self, n_class=10, input_dim=1, n_layers=1, hidden_dim=1000, drop_prob=0.2):
        super(ClsNetwork, self).__init__()
        assert n_layers >= 1
        self.n_class = n_class
        self.input_dim = input_dim + 1 # depthを含める
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.layers = nn.ModuleList()
        if self.n_layers == 1:
            self.layers.append(nn.Linear(self.input_dim*INPUT_WIDTH*INPUT_WIDTH, n_class*INPUT_WIDTH*INPUT_WIDTH))
        else:
            self.layers.append(nn.Linear(self.input_dim*INPUT_WIDTH*INPUT_WIDTH, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.drop_prob))
            for _ in range(self.n_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(self.drop_prob))
            self.layers.append(nn.Linear(hidden_dim, n_class*INPUT_WIDTH*INPUT_WIDTH))

    def forward(self, x):
        h = x
        batch_size = len(x)
        h = torch.reshape(h, (-1, self.input_dim*INPUT_WIDTH*INPUT_WIDTH))
        for layer in self.layers:
            h = layer(h)
        h = torch.reshape(h, (batch_size, self.n_class, INPUT_WIDTH, INPUT_WIDTH))
        return h
    
class RegNetwork(nn.Module):
    def __init__(self, input_dim=1, n_layers=1, hidden_dim=1000, drop_prob=0.2):
        super(RegNetwork, self).__init__()
        assert n_layers >= 1
        self.input_dim = input_dim + 1 # depthを含める
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.layers = nn.ModuleList()
        if self.n_layers == 1:
            self.layers.append(nn.Linear(self.input_dim*INPUT_WIDTH*INPUT_WIDTH, INPUT_WIDTH*INPUT_WIDTH))
        else:
            self.layers.append(nn.Linear(self.input_dim*INPUT_WIDTH*INPUT_WIDTH, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.drop_prob))
            for _ in range(self.n_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(self.drop_prob))
            self.layers.append(nn.Linear(hidden_dim, INPUT_WIDTH*INPUT_WIDTH))

    def forward(self, x):
        h = x
        batch_size = len(x)
        h = torch.reshape(h, (-1, self.input_dim*INPUT_WIDTH*INPUT_WIDTH))
        for layer in self.layers:
            h = layer(h)
        h = torch.reshape(h, (batch_size, INPUT_WIDTH, INPUT_WIDTH))
        return h