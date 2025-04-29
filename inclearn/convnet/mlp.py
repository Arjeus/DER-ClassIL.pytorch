# filepath: inclearn/convnet/mlp.py
import torch
from torch import nn

class MLP(nn.Module):
    """Simple MLP for tabular data like Iris dataset."""
    def __init__(self, input_dim, hidden_dim=64, remove_last_relu=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
