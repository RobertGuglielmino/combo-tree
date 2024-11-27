
import datetime
from pathlib import Path
import pickle

import numpy as np

from lib.classes.console_logger import ConsoleProgress


class DataManager:
    def __init__(self, data_dir="./processed_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.progress = ConsoleProgress()

    def save_processed_data(self, X, y, durations, metadata=None):
        """
        Save processed data with timestamp and metadata.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.data_dir / timestamp
        save_dir.mkdir(exist_ok=True)
        
        self.progress.print_header("Saving Processed Data")
        print(f"📂 Saving to: {save_dir}")

        # Save arrays
        try:
            np.save(save_dir / "X.npy", X)
            np.save(save_dir / "y.npy", y)
            np.save(save_dir / "durations.npy", durations)
            
            # Save metadata
            metadata = metadata or {}
            metadata.update({
                'timestamp': timestamp,
                'X_shape': X.shape,
                'y_shape': y.shape,
                'durations_shape': durations.shape,
                'data_types': {
                    'X': str(X.dtype),
                    'y': str(y.dtype),
                    'durations': str(durations.dtype)
                }
            })
            
            with open(save_dir / "metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            print("✅ Data saved successfully!")
            print(f"  • X shape: {X.shape}")
            print(f"  • y shape: {y.shape}")
            print(f"  • durations shape: {durations.shape}")
            
            return save_dir
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            raise

    def load_processed_data(self, data_dir=None):
        """
        Load processed data from directory.
        If no directory specified, loads the most recent.
        """
        try:
            if data_dir is None:
                # Get most recent directory
                dirs = sorted(Path(self.data_dir).glob("*"))
                if not dirs:
                    raise ValueError("No processed data found")
                data_dir = dirs[-1]
            
            self.progress.print_header("Loading Processed Data")
            print(f"📂 Loading from: {data_dir}")
            
            # Load arrays
            X = np.load(data_dir / "X.npy")
            y = np.load(data_dir / "y.npy")
            durations = np.load(data_dir / "durations.npy")
            
            # Load metadata
            with open(data_dir / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            print("✅ Data loaded successfully!")
            print(f"  • X shape: {X.shape}")
            print(f"  • y shape: {y.shape}")
            print(f"  • durations shape: {durations.shape}")
            print(f"  • Timestamp: {metadata['timestamp']}")
            
            return X, y, durations, metadata
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise