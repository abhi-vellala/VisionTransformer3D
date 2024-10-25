# Testing Custom ViT model

import torch
from transformers import ViTForImageClassification, ViTConfig, ViTModel

class ViTModelCustom(torch.nn.Module):
    def __init__(self, config, device) -> None:
        super(ViTModelCustom, self).__init__()
        self.num_labels = config.num_labels
        self.model = ViTModel(config).to(device)
        self.linear_layer = torch.nn.Linear(config.hidden_size, self.num_labels).to(device)
        self.softmax = torch.nn.Softmax(dim=-1).to(device)
        self.device = device

    def forward(self, x):
        x = self.model(x).last_hidden_state[:,0,:]
        # print(x)
        # # print(x.shape)
        # print(x[0])
        # print(x[0].shape)
        # # x = x[0].last_hidden_state
        # x = x.flatten().unsqueeze(0)
        x = self.linear_layer(x)
        x = self.softmax(x)
        return x
