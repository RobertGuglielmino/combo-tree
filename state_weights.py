import numpy as np
from typing import Dict

class StateWeights:
    """Weights for different components of the state vector"""
    def __init__(self):
        self.weights = {
            'position': {
                'p1_pos': 1.0,
                'p2_pos': 0.8,
                'relative_pos': 1.2  # Distance between characters
            },
            'damage': {
                'p1_damage': 0.6,
                'p2_damage': 0.6
            },
            'state': {
                'p1_state': 1.5,  # Action states are very important
                'p2_state': 1.2
            },
            'inputs': {
                'main_stick': 0.8,
                'c_stick': 0.6,
                'triggers': 0.4,
                'buttons': 1.0
            },
            'direction': {
                'p1_direction': 0.7,
                'p2_direction': 0.5
            }
        }
        
    def apply_weights(self, vector_components: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply weights to vector components and concatenate"""
        weighted_components = []
        
        # Position components
        weighted_components.extend(vector_components['p1_pos'] * self.weights['position']['p1_pos'])
        weighted_components.extend(vector_components['p2_pos'] * self.weights['position']['p2_pos'])
        weighted_components.extend(vector_components['relative_pos'] * self.weights['position']['relative_pos'])
        
        # Damage
        weighted_components.extend([
            vector_components['p1_damage'] * self.weights['damage']['p1_damage'],
            vector_components['p2_damage'] * self.weights['damage']['p2_damage']
        ])
        
        # States (one-hot encoded)
        weighted_components.extend(vector_components['p1_state'] * self.weights['state']['p1_state'])
        weighted_components.extend(vector_components['p2_state'] * self.weights['state']['p2_state'])
        
        # Inputs
        weighted_components.extend(vector_components['inputs']['main_stick'] * self.weights['inputs']['main_stick'])
        weighted_components.extend(vector_components['inputs']['c_stick'] * self.weights['inputs']['c_stick'])
        weighted_components.extend(vector_components['inputs']['triggers'] * self.weights['inputs']['triggers'])
        weighted_components.extend(vector_components['inputs']['buttons'] * self.weights['inputs']['buttons'])
        
        # Directions
        weighted_components.extend(vector_components['p1_direction'] * self.weights['direction']['p1_direction'])
        weighted_components.extend(vector_components['p2_direction'] * self.weights['direction']['p2_direction'])
        
        return np.concatenate([np.asarray(weighted_components, dtype="object")])
