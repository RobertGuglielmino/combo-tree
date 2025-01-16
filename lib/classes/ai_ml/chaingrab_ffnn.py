import torch
import torch.nn as nn

class ChainGrabFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=802):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Remove the sequence dimension since we don't need it anymore
        if len(x.shape) == 3:  # If input is [batch, sequence, features]
            x = x[:, -1, :]    # Take only the last state
        return self.network(x)