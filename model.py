from typing import List

import torch
from torch import nn
from torch.nn import Sequential, BatchNorm1d
from torch.nn.functional import relu
from torch.utils.data import TensorDataset, DataLoader


class MyModel(nn.Module):
    def __init__(self, mlps: List[int]):
        super().__init__()
        self.bn = Sequential()
        self.mlps = Sequential()
        for i in range(len(mlps) - 1):
            self.mlps.add_module(f"mlp{i}", nn.Linear(mlps[i], mlps[i + 1]))
            if i != len(mlps) - 2:
                self.bn.add_module(f"bn{i}", BatchNorm1d(mlps[i + 1]))

    def forward(self, x):
        for i, mlp in enumerate(self.mlps.children()):
            x = mlp(x)
            if i != len(self.mlps) - 1:
                x = relu(x)
                x = self.bn[i](x)
        return x


def get_dataloader(batch_size, *df_data, is_train=True):
    tensor_list = list()
    for i in range(len(df_data)):
        tensor_list.append(torch.tensor(df_data[i].values, dtype=torch.float32))
    my_dataset = TensorDataset(*tensor_list)
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=is_train)
    return my_dataloader
