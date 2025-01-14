import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lib.classes.ai_ml.model_loader import ModelLoader
from lib.classes.file_io.console_logger import ConsoleProgress
from lib.classes.nns.state_transition_preprocessor import StateTransitionPreprocessor
from lib.classes.ai_ml.transition_aware_ltsm import TransitionAwareLSTM


class SmashBrosAnalyzer:
    def __init__(self, rnn_units=64, feature_dim=802, max_memory_percent=0.2, epochs=1000, batch_size=256):
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

        
        self.rnn_lstm_model = TransitionAwareLSTM(
            input_dim=feature_dim,
            hidden_dim=rnn_units,
            output_dim=self._get_action_dim()
        ).to(self.device)

        self.scaler = torch.amp.GradScaler('cuda')
        
        self.optimizer = optim.AdamW(
            self.rnn_lstm_model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.preprocessor = StateTransitionPreprocessor(feature_dim, self.device)
        self.progress = ConsoleProgress()
        self.loader = ModelLoader("./training_models/my_model")

    def load_model(self, input):
        self.loader.load_model(input)
        
    def train_models(self, processed_data):
        (X, y, durations) = processed_data

        total_batches = len(X) // self.batch_size
        if len(X) % self.batch_size != 0:
            total_batches += 1

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=self.epochs,
            steps_per_epoch=total_batches,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        # set up logger
        start_time = time.time()
        self.progress.set_start_time(start_time)
        self.progress.set_epochs(self.epochs).set_total_samples(len(X)).set_batch_size(self.batch_size)
        self.progress.print_header("Training Model")
        self.progress.print_configuration()


        ### RUN MODEL ###

        for epoch in range(self.epochs):
            for batch_idx in range(total_batches):
                # Calculate batch indices
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(X))
                
                # Load single batch to GPU
                batch_X = self.get_chunk_float(X, start_idx, end_idx)
                batch_y = self.get_chunk_long(y, start_idx, end_idx)
                batch_durations = self.get_chunk_float(durations, start_idx, end_idx)

                try:
                    self.optimizer.zero_grad(set_to_none=True)

                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda'):
                        output = self.rnn_lstm_model(batch_X, batch_durations)
                        loss = self.criterion(output, batch_y)
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                    # Update progress
                    self.progress.update_status(loss, start_idx, epoch)

                finally:
                    # clean up
                    del batch_X, batch_y, batch_durations, output, loss
                    torch.cuda.empty_cache()

        # save
        self.loader.save_model(self.rnn_lstm_model.state_dict(), self.optimizer.state_dict(), self.feature_dim, self.rnn_units)

    
    def suggest_next_action(self, game_state, current_duration):
        if isinstance(game_state, list):
            game_state = np.array(game_state)
        
        sequence = np.tile(game_state, (self.preprocessor.sequence_length, 1)).reshape(1, self.preprocessor.sequence_length, -1)
        
        game_state = torch.FloatTensor(sequence).to(self.device)
        current_duration = torch.FloatTensor([current_duration]).to(self.device)
        
        self.rnn_lstm_model.eval()
        with torch.no_grad():
            logits = self.rnn_lstm_model(game_state, current_duration)
            probabilities = torch.softmax(logits, dim=1)
            prediction = probabilities.cpu().numpy()[0]
        
        action_idx = np.argmax(prediction)
        return action_idx, prediction[action_idx]
    

    def find_optimal_chunk_size(self, data_size):
        chunk_size = min(1024, data_size // 10)  # Start with 10% of data or 1024, whichever is smaller
        max_chunk = min(data_size, 10000)  # Cap maximum chunk size
        
        while chunk_size <= max_chunk:
            try:
                # Test memory with a sample chunk
                sample_data = torch.zeros(
                    (chunk_size, self.preprocessor.sequence_length, self.feature_dim), 
                    dtype=torch.float32, 
                    device=self.device
                )
                del sample_data
                torch.cuda.empty_cache()
                return chunk_size
            except RuntimeError:
                chunk_size = chunk_size // 2
                if chunk_size < 32:  # Minimum chunk size
                    raise RuntimeError("Unable to allocate even minimum chunk size")
                torch.cuda.empty_cache()

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def _get_action_dim(self):
        return 802  # Adjust based on the actual number of actions
    
    def get_chunk_long(self, data_list: np.ndarray, start, end):
        return torch.from_numpy(data_list[start:end]).long().to(self.device, non_blocking=True)
    
    def get_chunk_float(self, data_list: np.ndarray, start, end):
        return torch.from_numpy(data_list[start:end]).float().to(self.device, non_blocking=True)



