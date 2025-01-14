
import torch
import torch.nn as nn

class TransitionAwareLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super(TransitionAwareLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # LSTM expects input shape (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_dim + 1, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, durations):
        # Concatenate durations as an additional feature
        x = x.float()

        batch_size, seq_len, _ = x.shape
        durations = durations.float().unsqueeze(-1).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate input and durations
        x_with_duration = torch.cat([x, durations], dim=2)
        
        lstm_out, _ = self.lstm(x_with_duration)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out
