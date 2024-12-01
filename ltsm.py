# Required packages:
# numpy>=1.19.2
# scikit-learn>=0.24.2
# torch>=2.0.0
# pandas>=1.2.4

from pathlib import Path
import time
import numpy as np

from lib.classes.rnn_model import SmashBrosHybridAnalyzer
from lib.classes.replay_processor import ReplayProcessor
from lib.models.action_states import ACTION_STATES

# Example usage
def main():
    study()
    # predict()

def study():
    start_time = time.time()
    base_path = Path('C:/Users/Robert/CodingProjects/combo-tree')
    test_path = Path('E:/SlippiDataSet')

    analyzer = SmashBrosHybridAnalyzer(feature_dim=802)
    replays = ReplayProcessor(base_path, test_path, chunk_size=5)
        
    # Process Data
    replays.process_replays_RNN_chunked()
    
    # Train models
    analyzer.train_models()
    
    print("--- %s seconds ---" % (time.time() - start_time))



def predict():
    start_time = time.time()
    # Test prediction
    loaded_analyzer = SmashBrosHybridAnalyzer.load_model("./training_models/my_model.pt")
    test_state = np.array([1.0] * 20)  # Current state
    current_duration = 1  # How many frames this state has lasted
    action, confidence = loaded_analyzer.suggest_next_action(test_state, current_duration)
    print(f"Suggested action: {ACTION_STATES[action]}, Confidence: {confidence:.2f}")

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()