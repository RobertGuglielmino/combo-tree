
from typing import List, Tuple
import numpy as np
from lib.helpers.ensure_numpy import _ensure_numpy
import torch


class StateTransitionPreprocessor:
    def __init__(self, feature_dim: int, device):
        self.feature_dim = feature_dim
        self.state_map = {}
        self.next_index = 0
        self.sequence_length = 5
        self.device = device

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Ensures state has consistent shape: (feature_dim,)
        """
        if np.isscalar(state):
            return np.array([state])
        state = np.asarray(state)
        if state.ndim > 1:
            state = state.flatten()
        # Pad or truncate to feature_dim
        if len(state) < self.feature_dim:
            state = np.pad(state, (0, self.feature_dim - len(state)))
        elif len(state) > self.feature_dim:
            state = state[:self.feature_dim]
        return state

    def state_to_index(self, state) -> int:
        """
        Convert a state to a unique index.
        """
        state = self.normalize_state(state)
        state_key = tuple(state)
            
        if state_key not in self.state_map:
            self.state_map[state_key] = self.next_index
            self.next_index = (self.next_index + 1) % self._get_action_dim()
        
        return self.state_map[state_key]

    def detect_transitions(self, sequence: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """Detects state transitions in a sequence of frames."""
        transitions = []
        
        # Don't try to detect transitions in a single frame
        if len(sequence.shape) == 1 or sequence.shape[0] == 1:
            print("Warning: Single frame sequence detected")
            return transitions
            
        current_state = sequence[0]
        current_duration = 1
        
        for frame in sequence[1:]:
            if np.array_equal(frame, current_state):
                current_duration += 1
            else:
                transitions.append((current_state, current_duration))
                current_state = frame
                current_duration = 1
                
        transitions.append((current_state, current_duration))
        return transitions

    def preprocess_data(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_sequences = []
        all_targets = []
        all_durations = []
        
        # Combine sequences into larger chunks for transition detection
        sequence_buffer = []
        buffer_size = 20  # Adjust this based on your needs
        
        for sequence in data:
            if isinstance(sequence, np.ndarray) and sequence.ndim == 1:
                sequence_buffer.append(sequence)
                
                if len(sequence_buffer) >= buffer_size:
                    # Process buffer as one sequence
                    combined_sequence = np.stack(sequence_buffer)
                    transitions = self.detect_transitions(combined_sequence)
                    
                    if len(transitions) >= 2:
                        for i in range(len(transitions) - 1):
                            current_state, duration = transitions[i]
                            next_state = transitions[i+1][0]
                            
                            sequence_array = np.tile(self.normalize_state(current_state), 
                                                  (self.sequence_length, 1))
                            next_state_idx = self.state_to_index(next_state)
                            
                            all_sequences.append(sequence_array)
                            all_targets.append(next_state_idx)
                            all_durations.append(duration)
                    
                    sequence_buffer = []
        
        # TODO can combine these two

        # Process remaining buffer
        if sequence_buffer:
            combined_sequence = np.stack(sequence_buffer)
            transitions = self.detect_transitions(combined_sequence)
            if len(transitions) >= 2:
                for i in range(len(transitions) - 1):
                    current_state, duration = transitions[i]
                    next_state = transitions[i+1][0]
                    
                    sequence_array = np.tile(self.normalize_state(current_state), 
                                          (self.sequence_length, 1))
                    next_state_idx = self.state_to_index(next_state)
                    
                    all_sequences.append(sequence_array)
                    all_targets.append(next_state_idx)
                    all_durations.append(duration)
        
        if not all_sequences:
            raise ValueError("No valid sequences found in data")
            
        return (_ensure_numpy(torch.from_numpy(np.array(all_sequences, dtype=np.float32)).float().to(self.device)),
                _ensure_numpy(torch.from_numpy(np.array(all_targets, dtype=np.int64)).long().to(self.device)),
                _ensure_numpy(torch.from_numpy(np.array(all_durations, dtype=np.float32)).float().to(self.device)))
    
    def _get_action_dim(self) -> int:
        return 20