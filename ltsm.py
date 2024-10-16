import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl

class GameStateVector:
    def __init__(self):
        # Constants for categorical encodings
        self.ACTION_STATES = {
            'STANDING': 0,
            'DASHING': 1,
            'RUNNING': 2,
            'JUMPING': 3,
            'FALLING': 4,
            'ATTACKING': 5,
            # ... add all relevant action states
        }
        
        self.MOVES = {
            'NEUTRAL_B': 0,
            'SIDE_B': 1,
            'UP_B': 2,
            'DOWN_B': 3,
            'NAIR': 4,
            'FAIR': 5,
            # ... add all possible moves
        }
        
        self.scaler = StandardScaler()
        
    def create_state_vector(self, frame_data):
        """Convert a single frame of Slippi data into a normalized vector"""
        
        # 1. Position and Motion (continuous values)
        position_vector = [
            frame_data.player.x,
            frame_data.player.y,
            frame_data.opponent.x,
            frame_data.opponent.y,
            frame_data.player.x_velocity,
            frame_data.player.y_velocity,
            frame_data.opponent.x_velocity,
            frame_data.opponent.y_velocity
        ]
        
        # 2. Damage and Stocks (continuous values)
        damage_vector = [
            frame_data.player.percent,
            frame_data.opponent.percent,
            frame_data.player.stocks,
            frame_data.opponent.stocks
        ]
        
        # 3. Action States (one-hot encoded)
        player_action = np.zeros(len(self.ACTION_STATES))
        player_action[self.ACTION_STATES[frame_data.player.action_state]] = 1
        
        opponent_action = np.zeros(len(self.ACTION_STATES))
        opponent_action[self.ACTION_STATES[frame_data.opponent.action_state]] = 1
        
        # 4. Controller Inputs (continuous values)
        input_vector = [
            frame_data.player.controller.main_stick_x,
            frame_data.player.controller.main_stick_y,
            frame_data.player.controller.c_stick_x,
            frame_data.player.controller.c_stick_y,
            frame_data.player.controller.l_trigger,
            frame_data.player.controller.r_trigger
        ]
        
        # 5. Stage Info (normalized coordinates)
        stage_vector = [
            frame_data.stage.platform1_x,
            frame_data.stage.platform1_y,
            frame_data.stage.platform2_x,
            frame_data.stage.platform2_y,
            frame_data.stage.blast_zone_right,
            frame_data.stage.blast_zone_left,
            frame_data.stage.blast_zone_top,
            frame_data.stage.blast_zone_bottom
        ]
        
        # Combine all vectors
        complete_vector = np.concatenate([
            position_vector,
            damage_vector,
            player_action,
            opponent_action,
            input_vector,
            stage_vector
        ])
        
        return complete_vector
    
    def create_sequence_vector(self, replay_data, sequence_length=60):
        """Create a sequence of state vectors from multiple frames"""
        sequence = []
        
        for frame in range(sequence_length):
            if frame < len(replay_data):
                frame_vector = self.create_state_vector(replay_data[frame])
            else:
                # Pad with zeros if sequence is shorter than desired length
                frame_vector = np.zeros(self.get_vector_size())
            sequence.append(frame_vector)
            
        return np.array(sequence)

class MeleeDataset(Dataset):
    def __init__(self, replay_files, sequence_length=60):
        self.vectorizer = GameStateVector()
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        
        for replay in replay_files:
            self.process_replay(replay)
    
    def process_replay(self, replay):
        # Process each decision point in the replay
        for i in range(len(replay) - self.sequence_length):
            sequence = self.vectorizer.create_sequence_vector(
                replay[i:i+self.sequence_length]
            )
            next_action = replay[i+self.sequence_length].player.action
            
            self.sequences.append(sequence)
            self.labels.append(self.vectorizer.MOVES[next_action])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class MeleeLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_actions):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # Take only last sequence output
        return self.fc(out)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Example usage
def train_model(replay_files):
    # Create dataset
    dataset = MeleeDataset(replay_files)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    input_size = dataset.vectorizer.get_vector_size()
    model = MeleeLSTM(
        input_size=input_size,
        hidden_size=256,
        num_actions=len(dataset.vectorizer.MOVES)
    )
    
    # Train model
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, train_loader)
    
    return model