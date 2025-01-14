from pathlib import Path
import time
import numpy as np

from lib.classes.ai_ml.smash_bros_analyzer import SmashBrosAnalyzer
from lib.classes.file_io.replay_processor import ReplayProcessor
from lib.models.action_states import ACTION_STATES

# Example usage
def main():
    study()
    # predict()

def study():
    start_time = time.time()
    base_path = Path('C:/Users/Robert/CodingProjects/combo-tree')
    test_path = Path('E:/SlippiDataSet')
    base_path = Path('E:/SlippiDataSet')

    analyzer = SmashBrosAnalyzer(feature_dim=100)
    replays = ReplayProcessor(base_path, test_path, chunk_size=5)
        
    # Process Data
    processed_data = replays.process_replays_RNN_chunked()
    
    # Train models
    analyzer.train_models(processed_data)
    
    print("--- %s seconds ---" % (time.time() - start_time))



def predict():
    start_time = time.time()
    # Test prediction
    loaded_analyzer = SmashBrosAnalyzer.load_model("./training_models/my_model.pt")
    test_state = np.array([1.0] * 20)  # Current state
    current_duration = 1  # How many frames this state has lasted
    action, confidence = loaded_analyzer.suggest_next_action(test_state, current_duration)
    print(f"Suggested action: {ACTION_STATES[action]}, Confidence: {confidence:.2f}")

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()