import time
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from annoy import AnnoyIndex
from peppi_py import read_slippi
import concurrent.futures
from tqdm import tqdm
import pickle
from game_states import GameState, _frame_to_game_state, _game_state_to_vector, _get_matchup_key, _get_next_significant_action, flatten_and_convert_vector, print_action_sequence
from lib.models.action_states import ACTION_STATES
from lib.helpers.matchup_key import MatchupKey

class ParallelTransitionMatcher:
    def __init__(self, base_path: Path, vector_dim: int = 802):
        self.base_path = base_path
        self.vector_dim = vector_dim
        self.matchup_indices: Dict[MatchupKey, AnnoyIndex] = {}
        self.matchup_caches: Dict[MatchupKey, Dict[int, str]] = {}

    # input:   a replay file
    # output:  a dictionary of the replays Character Matchup (fox vs. marth) -> all the frames of the replay as normalized vectors
    def process_replay(self, replay_file: Path) -> Dict[MatchupKey, List[Tuple[np.ndarray, str]]]:
        try:
            game = read_slippi(str(replay_file))
            matchup = _get_matchup_key(game)
            results = {}

            # Process from P1's perspective
            results[matchup] = self._process_replay_perspective(game, is_p1_perspective=True)

            # If it's not a mirror match, process from P2's perspective
            if not matchup.is_mirror:
                reversed_matchup = matchup.reversed
                results[reversed_matchup] = self._process_replay_perspective(game, is_p1_perspective=False)

            return results
        except Exception as e:
            print(f"Error reading and processing slippi file {replay_file}: {e}")
            return {}

    def _process_replay_perspective(self, game, is_p1_perspective: bool) -> List[Tuple[np.ndarray, str]]:
        processed_states = []
        length = game.metadata["lastFrame"]
        
        i = 0

        while i < length:
            initial_state = _frame_to_game_state(game.frames, game.start, i, is_p1_perspective)
            state_vector = _game_state_to_vector(initial_state)

            action_sequence = _get_next_significant_action(game, i, is_p1_perspective)
            processed_states.append((state_vector, action_sequence))

            # Skip to the frame after the final action
            i += len(action_sequence.intermediary_states) + 1

        return processed_states

    def _compute_average_state_vector(self, states: List[GameState]) -> np.ndarray:
        vectors = [_game_state_to_vector(state) for state in states]
        return np.mean(vectors, axis=0)


    def build_indices(self, replay_files: List[str], n_trees: int = 10):
        all_results = {}
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(self.process_replay, replay_file): replay_file for replay_file in replay_files}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(replay_files), desc="Processing replays"):
                replay_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        for matchup, states in result.items():
                            if matchup not in all_results:
                                all_results[matchup] = []
                            all_results[matchup].extend(states)
                except Exception as e:
                    print(f"Error processing replay file: {replay_file}: {e}")

        self._update_indices(all_results)

        for index in self.matchup_indices.values():
            index.build(n_trees)

        self.save_indices()

    def _update_indices(self, result: Dict[MatchupKey, List[Tuple[np.ndarray, str]]]):
        for matchup, states in result.items():
            if matchup not in self.matchup_indices:
                # Initialize the index with the correct dimension after flattening
                sample_vector = flatten_and_convert_vector(states[0][0])
                if self.vector_dim is None:
                    self.vector_dim = len(sample_vector)
                self.matchup_indices[matchup] = AnnoyIndex(self.vector_dim, 'euclidean')
                self.matchup_caches[matchup] = {}

            index = self.matchup_indices[matchup]
            cache = self.matchup_caches[matchup]
            current_index_size = len(cache)


            for i, (state_vector, next_action) in enumerate(states):
                flattened_vector = flatten_and_convert_vector(state_vector)
                
                # Ensure the vector has the correct dimension
                if len(flattened_vector) != self.vector_dim:
                    print(f"Warning: Vector dimension mismatch. Expected {self.vector_dim}, got {len(flattened_vector)}")
                    continue  # Skip this vector or handle the error as appropriate

                index.add_item(current_index_size + i, state_vector)
                cache[current_index_size + i] = next_action

    def save_indices(self):
        for matchup, index in self.matchup_indices.items():
            matchup_path = self.base_path / f"{matchup.p1_char}_vs_{matchup.p2_char}"
            matchup_path.mkdir(parents=True, exist_ok=True)
            
            index.save(str(matchup_path / "index.ann"))
            with open(matchup_path / "cache.pkl", 'wb') as f:
                pickle.dump(self.matchup_caches[matchup], f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_indices(self):
        """Load all available matchup indices"""
        for matchup_path in self.base_path.glob("*_vs_*"):
            try:
                p1_char, p2_char = map(str, matchup_path.name.split("_vs_"))
                matchup = MatchupKey(p1_char, p2_char)
                
                index = AnnoyIndex(self.vector_dim, 'euclidean')
                index.load(str(matchup_path / "index.ann"))
                self.matchup_indices[matchup] = index
                
                with open(matchup_path / "cache.pkl", 'rb') as f:
                    self.matchup_caches[matchup] = pickle.load(f)
                    
            except (ValueError, FileNotFoundError) as e:
                print(f"Error loading matchup {matchup_path}: {e}")


    def find_similar_states(self, current_state: GameState, matchup: MatchupKey, n_neighbors: int = 10) -> List[Tuple[str, float]]:

        if matchup not in self.matchup_indices:
            raise ValueError(f"No data available for matchup: {matchup}")

        index = self.matchup_indices[matchup]
        cache = self.matchup_caches[matchup]
        # print(cache)

        state_vector = _game_state_to_vector(current_state)
        indices, distances = index.get_nns_by_vector(state_vector, n_neighbors, include_distances=True)

        return [(cache[idx], dist) for idx, dist in zip(indices, distances)]
    

def study():
    start_time = time.time()
    base_path = Path("C:\\Users\\Robert\\CodingProjects\\combo-tree\\lib\\test\\test replays")
    matcher = ParallelTransitionMatcher(base_path)

    replay_files = list(base_path.glob("*.slp"))
    print(f"Found {len(replay_files)} replay files")

    matcher.build_indices(replay_files)
    print("--- %s seconds ---" % (time.time() - start_time))

def predict():
    base_path = Path("C:\\Users\\Robert\\CodingProjects\\combo-tree\\lib\\test\\test replays")
    matcher = ParallelTransitionMatcher(base_path)
    matcher.load_indices()
    test_game = read_slippi('./Game_20220910T180650.slp')
    test_frame = 460
    print(ACTION_STATES[test_game.frames.ports[1].leader.pre.state[test_frame].as_py()])

    # # Example of using the matcher
    similar_states = matcher.find_similar_states(_frame_to_game_state(test_game.frames, test_game.start, test_frame, True), _get_matchup_key(test_game))
    print("Similar states and their next actions:")
    for next_action, distance in similar_states:
        print(f"Next action: {print_action_sequence(next_action)}, Distance: {distance:.3f}")

# Example usage
def main():
    study()
    predict()

if __name__ == "__main__":
    main()