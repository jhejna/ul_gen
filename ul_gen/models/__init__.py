import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers=[], activation_fn="relu", bias=True):
        super().__init__()
        if activation_fn == 'relu':
            self.activation = nn.ReLU
        elif activation_fn == 'leaky':
            self.activation = lambda: nn.LeakyReLU(0.2)
        
        model = []
        for size in hidden_layers:
            model.append(nn.Linear(input_size, size, bias))
            model.append(self.activation())
            input_size = size
        model.append(nn.Linear(input_size, output_size, bias))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

