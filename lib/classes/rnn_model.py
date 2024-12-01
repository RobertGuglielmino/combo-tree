
import sys
import time
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lib.classes.console_logger import ConsoleProgress
from lib.classes.state_transition_preprocessor import StateTransitionPreprocessor
from lib.classes.transition_aware_ltsm import TransitionAwareLSTM
from lib.classes.replay_processor import ReplayProcessor


class SmashBrosHybridAnalyzer:
    def __init__(self, rnn_units=64, feature_dim=802, max_memory_percent=0.7, epochs=10, batch_size=256):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        self.feature_dim = feature_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.rnn_units = rnn_units

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            # Limit GPU memory usage
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(max_memory_percent)
            torch.cuda.empty_cache()
        
        self.rnn_model = TransitionAwareLSTM(
            input_dim=feature_dim,
            hidden_dim=rnn_units,
            output_dim=self._get_action_dim()
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.rnn_model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.preprocessor = StateTransitionPreprocessor(feature_dim, self.device)
        self.replays = ReplayProcessor(feature_dim=self.feature_dim)
        self.progress = ConsoleProgress()


    def save_model(self, filepath: str):
        model_state = {
            'model_state_dict': self.rnn_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_dim': self.feature_dim,
            'rnn_units': self.rnn_units,
            'preprocessor_state': {
                'state_map': self.preprocessor.state_map,
                'next_index': self.preprocessor.next_index,
                'feature_dim': self.preprocessor.feature_dim,
                'sequence_length': self.preprocessor.sequence_length
            }
        }
        
        try:
            torch.save(model_state, f"{filepath}.pt")
            print(f"Model saved successfully to {filepath}.pt")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    @classmethod
    def load_model(cls, filepath: str):
        try:
            # Load the saved state
            checkpoint = torch.load(filepath)
            
            # Create new instance with saved parameters
            instance = cls(
                feature_dim=checkpoint['feature_dim'],
                rnn_units=checkpoint['rnn_units']
            )
            
            # Load model state
            instance.rnn_model.load_state_dict(checkpoint['model_state_dict'])
            instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load preprocessor state
            preprocessor_state = checkpoint['preprocessor_state']
            instance.preprocessor.state_map = preprocessor_state['state_map']
            instance.preprocessor.next_index = preprocessor_state['next_index']
            instance.preprocessor.feature_dim = preprocessor_state['feature_dim']
            instance.preprocessor.sequence_length = preprocessor_state['sequence_length']
            
            print(f"Model loaded successfully from {filepath}")
            return instance
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        
    def train_models(self):
        chunk_size = 100  # Adjust based on your memory constraints
        self.progress.set_epochs(self.epochs).set_total_samples(len(self.processed_X)).set_batch_size(self.batch_size)
        self.progress.print_header("Training Model")
        self.progress.print_configuration()

        for i in range(0, len(self.processed_X), chunk_size):
            chunk_X = self.get_chunk_float(self.processed_X, i, chunk_size)
            chunk_y = self.get_chunk_long(self.processed_y, i, chunk_size)
            chunk_durations = self.get_chunk_float(self.processed_durations, i, chunk_size)
            
            self.run_model(chunk_X, chunk_y, chunk_durations)
            
            del chunk_X, chunk_y, chunk_durations
            torch.cuda.empty_cache()

        
        self.save_model("./training_models/my_model")


    def run_model(self, X_train, y_train, durations): 

        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
        if not isinstance(durations, torch.Tensor):
            durations = torch.tensor(durations, dtype=torch.float32)
            
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        durations = durations.to(self.device)
        start_time = time.time()
        self.progress \
            .set_start_time(start_time)

        for epoch in range(self.epochs):

            for j in range(0, X_train.shape[0], self.batch_size):

                batch_X = X_train[j:j+self.batch_size]
                batch_y = y_train[j:j+self.batch_size]
                batch_durations = durations[j:j+self.batch_size]
                
                self.optimizer.zero_grad()
                output = self.rnn_model(batch_X, batch_durations)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()

                self.progress.update_status(loss, j, epoch)
                
                
                # Clean up batch tensors
                del output, loss
                torch.cuda.empty_cache()
            
        # Clear memory after processing each chunk
        del X_train, y_train, durations
        torch.cuda.empty_cache()
    
    def suggest_next_action(self, game_state, current_duration):
        if isinstance(game_state, list):
            game_state = np.array(game_state)
        
        sequence = np.tile(game_state, (self.preprocessor.sequence_length, 1)).reshape(1, self.preprocessor.sequence_length, -1)
        
        game_state = torch.FloatTensor(sequence).to(self.device)
        current_duration = torch.FloatTensor([current_duration]).to(self.device)
        
        self.rnn_model.eval()
        with torch.no_grad():
            logits = self.rnn_model(game_state, current_duration)
            probabilities = torch.softmax(logits, dim=1)
            prediction = probabilities.cpu().numpy()[0]
        
        action_idx = np.argmax(prediction)
        return action_idx, prediction[action_idx]
    



    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def _get_action_dim(self):
        return 802  # Adjust based on the actual number of actions
    
    def get_chunk_long(self, data_list: np.ndarray, index, offset):
        return torch.from_numpy(data_list[index:index+offset]).long().to(self.device)
    
    def get_chunk_float(self, data_list: np.ndarray, index, offset):
        return torch.from_numpy(data_list[index:index+offset]).float().to(self.device)



