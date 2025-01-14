
import torch

class ModelLoader:
    def __init__(self, filepath = '.'):
        self.filepath = filepath        
    
    def save_model(self, model_state_dict, optimizer_state, feature_dim, rnn_units):
        model_state = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state,
            'feature_dim': feature_dim,
            'rnn_units': rnn_units
        }
        
        try:
            torch.save(model_state, f"{self.filepath}.pt")
            print(f"Model saved successfully to {self.filepath}.pt")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    @classmethod
    def load_model(self, cls):
        try:
            # Load the saved state
            checkpoint = torch.load(self.filepath)
            
            # Create new instance with saved parameters
            instance = cls(
                feature_dim=checkpoint['feature_dim'],
                rnn_units=checkpoint['rnn_units']
            )
            
            # Load model state
            instance.rnn_lstm_model.load_state_dict(checkpoint['model_state_dict'])
            instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Model loaded successfully from {self.filepath}")
            return instance
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise



# DONT KNOW IF WE NEED

# , 
#             'preprocessor_state': {
#                 'state_map': self.preprocessor.state_map,
#                 'next_index': self.preprocessor.next_index,
#                 'feature_dim': self.preprocessor.feature_dim,
#                 'sequence_length': self.preprocessor.sequence_length
#             }



            # # Load preprocessor state
            # preprocessor_state = checkpoint['preprocessor_state']
            # instance.preprocessor.state_map = preprocessor_state['state_map']
            # instance.preprocessor.next_index = preprocessor_state['next_index']
            # instance.preprocessor.feature_dim = preprocessor_state['feature_dim']
            # instance.preprocessor.sequence_length = preprocessor_state['sequence_length']