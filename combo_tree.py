import time
import numpy as np
from typing import Any, List, Dict, Tuple
from pathlib import Path
from annoy import AnnoyIndex
from peppi_py import read_slippi
import concurrent.futures
from tqdm import tqdm
import pickle
from lib.models.characters_by_id import CHARACTERS_BY_ID
from game_states import GameState, _frame_to_game_state, _game_state_to_vector
from lib.models.action_states import ACTION_STATES
from lib.helpers.matchup_key import MatchupKey


def _get_matchup_key(game) -> MatchupKey:
    active_ports = list(filter(lambda x: x, game.start.players))
    p1_char = CHARACTERS_BY_ID[active_ports[0].character]
    p2_char = CHARACTERS_BY_ID[active_ports[1].character]
    return MatchupKey(p1_char, p2_char)

def flatten_and_convert_vector(vector: Any) -> List[float]:
    """
    Recursively flatten and convert a potentially nested structure into a list of floats.
    """
    def flatten_recursive(item):
        if isinstance(item, (int, float)):
            return [float(item)]
        elif isinstance(item, np.ndarray):
            return flatten_recursive(item.tolist())
        elif isinstance(item, (list, tuple)):
            return [subitem for i in item for subitem in flatten_recursive(i)]
        else:
            raise ValueError(f"Unsupported type in vector: {type(item)}")

    try:
        flattened = flatten_recursive(vector)
        return flattened
    except Exception as e:
        print(f"Error flattening vector: {e}")
        print(f"Problematic vector: {vector}")
        raise
    

class ParallelTransitionMatcher:
    def __init__(self, base_path: Path, vector_dim: int = 772):
        self.base_path = base_path
        self.vector_dim = vector_dim
        self.matchup_indices: Dict[MatchupKey, AnnoyIndex] = {}
        self.matchup_caches: Dict[MatchupKey, Dict[int, str]] = {}

    def process_replay(self, replay_file: Path) -> Dict[MatchupKey, List[Tuple[np.ndarray, str]]]:
        try:
            game = read_slippi(str(replay_file))
            matchup = _get_matchup_key(game)
            processed_states = []

            current_action = None
            action_frames = []
            state = None
            next_state = None

            for i in range(len(game.frames.id) - 1):
                if state:
                    state = next_state
                else:
                    state = _frame_to_game_state(game.frames, game.start, i)
                next_state = _frame_to_game_state(game.frames, game.start, i + 1)

                if state.p1_state != current_action:
                    if current_action is not None:
                        avg_state_vector = self._compute_average_state_vector(action_frames)
                        next_action = next_state.p1_state
                        processed_states.append((avg_state_vector, next_action))
                    
                    current_action = state.p1_state
                    action_frames = []

                action_frames.append(state)

            if action_frames:
                avg_state_vector = self._compute_average_state_vector(action_frames)
                next_action = "END_OF_GAME"
                processed_states.append((avg_state_vector, next_action))

            return {matchup: processed_states}
        except Exception as e:
            print(e)
            print(f"Error reading and processing slippi file {replay_file}: {e}")
            return {}

    def _compute_average_state_vector(self, states: List[GameState]) -> np.ndarray:
        vectors = [_game_state_to_vector(state) for state in states]
        return np.mean(vectors, axis=0)

    def build_indices(self, replay_files: List[Path], n_trees: int = 10):
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(self.process_replay, replay_file): replay_file for replay_file in replay_files}
            
            print("process_replay")
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(replay_files), desc="Processing replays"):
                replay_file = future_to_file[future]
                # try:
                result = future.result()
                if result:
                    results.append(result)
                # except Exception as e:
                #     print(f"Error completing future and appending file: {replay_file}: {e}")

        for result in results:
            self._update_indices(result)

        for matchup, index in self.matchup_indices.items():
            print(f"Building index for matchup: {matchup}")
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
                
                print(f"Original state vector: {len(state_vector)}")
                flattened_vector = flatten_and_convert_vector(state_vector)
                print(f"Flattened vector: {len(flattened_vector)}")
                
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
    test_frame = 4876
    print(ACTION_STATES[test_game.frames.ports[0].leader.pre.state[test_frame].as_py()])

    # # Example of using the matcher
    similar_states = matcher.find_similar_states(_frame_to_game_state(test_game.frames, test_game.start, test_frame), _get_matchup_key(test_game))
    print("Similar states and their next actions:")
    for next_action, distance in similar_states:
        print(f"Next action: {ACTION_STATES[next_action]}, Distance: {distance:.3f}")


# Example usage
def main():
    study()
    predict()

if __name__ == "__main__":
    main()