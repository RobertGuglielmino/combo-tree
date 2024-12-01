import time
from typing import Dict, List, Tuple
import concurrent
import numpy as np
import psutil
from game_states import _frame_to_vector, _get_next_significant_action
from lib.helpers.matchup_key import MatchupKey, _get_matchup_key
from peppi_py import read_slippi

from lib.helpers.update_progress import update_progress

class ReplayProcessor:
    def __init__(self, replay_path, test_replay_path="/", chunk_size=5, feature_dim=100):
        self.replays = list(replay_path.glob("*.slp"))
        self.test_replays = list(test_replay_path.glob("*.slp"))
        self.chunk_size = chunk_size
        self.feature_dim=feature_dim

        print(f"Found {len(self.replays)} replay files")
        print(f"===============================")
        
        self.processed_X = np.empty((0, 5, self.feature_dim), dtype=np.float32)  # (samples, sequence_length, features)
        self.processed_y = np.empty(0, dtype=np.int64)  # (samples,)
        self.processed_durations = np.empty(0, dtype=np.float32)  # (samples,)


    def process_replays_NNS(self, replay_files):
        start_time = time.time()
        num_files = len(replay_files)
    
        print(f"Starting to process {num_files} replay files...")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process files with simple progress indicator
            results = []
            for i, result in enumerate(executor.map(self._single_replay_to_normalized_vector, replay_files)):
                if i % 10 == 0 or i == num_files - 1:  # Update more frequently
                    update_progress(i + 1, num_files, start_time)
                results.append(result)

        return results

    # TODO _process_replay_perspective_with_sequence now returns Tuple(List, List),
    # remove boolean from params somehow

    # output:  a dictionary of the replays Character Matchup (fox vs. marth) -> all the frames of the replay as normalized vectors
    def _single_replay_to_normalized_vector(self, replay_file) -> Dict[MatchupKey, List[Tuple[np.ndarray, str]]]:
            try:
                # cache_file = os.path.join(self.cache_dir, f"{os.path.basename(str(replay_file))}.pickle")
            
                # if os.path.exists(cache_file):
                #     print("game")
                #     print(cache_file)
                #     with open(cache_file, 'rb') as f:
                #         return pickle.load(f)

                game = read_slippi(str(replay_file))

                matchup = _get_matchup_key(game)
                results = {}

                results[matchup] = self._vector_sequence_from_perspective(replay_file, mirror=True)

                reversed_matchup = matchup.reversed
                results[reversed_matchup] = self._vector_sequence_from_perspective(replay_file, is_p1_perspective=False)

                # with open(cache_file, 'wb') as f:
                #     pickle.dump(results, f)
                return results
            
            except Exception as e:
                print(f"Error reading and processing slippi file {replay_file}: {e}")
                return {}


    def process_replays_RNN_chunked(self):
    # Process replays in chunks
        for chunk_data in self._RNN_replay_chunk():
            try:
                self.add_replays_batch(chunk_data)
                
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
            
            finally:
                del chunk_data
                import gc
                gc.collect()

                
    def add_replays_batch(self, replay_vectors: List[np.ndarray]):
        """Process and store data immediately in the correct format"""
        X, y, durations = self.preprocessor.preprocess_data(replay_vectors)

        # Extend our stored preprocessed data
        self.processed_X = np.concatenate([self.processed_X, X], axis=0)
        self.processed_y = np.concatenate([self.processed_y, y], axis=0)
        self.processed_durations = np.concatenate([self.processed_durations, durations], axis=0)
    

    def _RNN_replay_chunk(self):
        for i in range(0, len(self.replays), self.chunk_size):
            chunk = self.replays[i:i + self.chunk_size]
            chunk_data = []
            print ("\033[A                                                                                          \033[A")
            print(f"\rProcessing replays {i+1} to {min(i+self.chunk_size, len(self.replays))} of {len(self.replays)} ---- Current RAM memory % usage: {psutil.virtual_memory()[2]}")
            
            for file in chunk:
                try:
                    states, _ = self._vector_sequence_from_perspective(file, character=18)
                    chunk_data.extend(states)
                    del file
                    
                except Exception as e:
                    print(f"Error processing RNN chunk at file {file}: {e}")
                    continue
                
            yield chunk_data
            del chunk_data

    def _vector_sequence_from_perspective(self, replay_file_path, character=None, mirror=False) -> Tuple[List[np.ndarray], List[str]]:
        game = read_slippi(str(replay_file_path))
        print ("\033[A                             \033[A")

        port = 0
        processed_states = []
        action_sequences = []
        frame = 0

        if not character:
            if game.start.players[1].character == character:
                port = 1

        #if not mirror
            
        while frame < game.metadata["lastFrame"]:
            state_vector = _frame_to_vector(game, frame, port)
            action_sequence = _get_next_significant_action(game, frame, port)
            frame += len(action_sequence.intermediary_states) + 1

            processed_states.append(state_vector)
            action_sequences.append(action_sequence)

        return processed_states, action_sequences
