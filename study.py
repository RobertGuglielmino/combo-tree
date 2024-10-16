import os
import time
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from annoy import AnnoyIndex
from peppi_py import read_slippi
import concurrent.futures
from tqdm import tqdm
import pickle
from game_states import ActionSequence, _compress_states, _frame_to_vector, _get_next_significant_action
from lib.helpers.matchup_key import MatchupKey, _get_matchup_key

class ParallelTransitionStateIndexer:
    def __init__(self, base_path: Path, vector_dim: int = 802, intermediary_states: List[str] = [], cache_dir: str = "cache"):
        self.base_path = base_path
        self.vector_dim = vector_dim
        self.matchup_indices: Dict[MatchupKey, AnnoyIndex] = {}
        self.matchup_caches: Dict[MatchupKey, Dict[int, ActionSequence]] = {}
        self.intermediary_states = set(intermediary_states)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def process_and_update(self, replay_file):
        try:
            result = self.process_replay(replay_file)
            if result:
                return result
        except Exception as e:
            print(f"Error processing replay file: {replay_file}: {e}")
        return None

    def build_indices(self, replay_files: List[str], n_trees: int = 10):
        all_results = {}
        
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.process_and_update, replay_files), total=len(replay_files), desc="Processing replays"))
        
        for result in results:
            if result:
                for matchup, states in result.items():
                    if matchup not in all_results:
                        all_results[matchup] = []
                    all_results[matchup].extend(states)

        self._update_indices(all_results)

        for index in self.matchup_indices.values():
            index.build(n_trees)

        self.save_indices()

    # input:   a replay file
    # output:  a dictionary of the replays Character Matchup (fox vs. marth) -> all the frames of the replay as normalized vectors
    def process_replay(self, replay_file: Path) -> Dict[MatchupKey, List[Tuple[np.ndarray, str]]]:
        try:
            cache_file = os.path.join(self.cache_dir, f"{os.path.basename(str(replay_file))}.pickle")
        
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

            game = read_slippi(str(replay_file))

            matchup = _get_matchup_key(game)
            results = {}

            # Process from P1's perspective
            results[matchup] = self._process_replay_perspective(game, is_p1_perspective=True)

            # If it's not a mirror match, process from P2's perspective
            if not matchup.is_mirror:
                reversed_matchup = matchup.reversed
                results[reversed_matchup] = self._process_replay_perspective(game, is_p1_perspective=False)

            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)

            return results
        
        except Exception as e:
            print(f"Error reading and processing slippi file {replay_file}: {e}")
            return {}

    def _process_replay_perspective(self, game, is_p1_perspective: bool) -> List[Tuple[np.ndarray, str]]:
        processed_states = []
        i = 0

        while i < game.metadata["lastFrame"]:
            state_vector = _frame_to_vector(game, i, is_p1_perspective)

            action_sequence = _get_next_significant_action(game, i, is_p1_perspective)
            processed_states.append((state_vector, action_sequence))

            # Skip to the frame after the final action
            i += len(action_sequence.intermediary_states) + 1

        return processed_states


    def _update_indices(self, results: Dict[MatchupKey, List[Tuple[np.ndarray, ActionSequence]]]):
        for matchup, states in results.items():
            compressed_states = _compress_states(states)
            self._process_matchup_states(matchup, compressed_states)

    def _process_matchup_states(self, matchup: MatchupKey, states: List[Tuple[np.ndarray, ActionSequence]]):
        index, cache = self._get_or_create_index(matchup)
        current_index_size = len(cache)

        for i, (state_vector, action_sequence) in enumerate(states):
            index.add_item(current_index_size + i, state_vector)
            cache[current_index_size + i] = action_sequence

    def _get_or_create_index(self, matchup: MatchupKey) -> Tuple[AnnoyIndex, Dict]:
        if matchup not in self.matchup_indices:
            self.matchup_indices[matchup] = AnnoyIndex(self.vector_dim, 'euclidean')
            self.matchup_caches[matchup] = {}
        return self.matchup_indices[matchup], self.matchup_caches[matchup]
    

    def save_indices(self):
        for matchup, index in self.matchup_indices.items():
            matchup_path = self.base_path / f"{matchup.p1_char}_vs_{matchup.p2_char}"
            matchup_path.mkdir(parents=True, exist_ok=True)
            
            index.save(str(matchup_path / "index.ann"))
            with open(matchup_path / "cache.pkl", 'wb') as f:
                pickle.dump(self.matchup_caches[matchup], f, protocol=pickle.HIGHEST_PROTOCOL)
    
def study():
    start_time = time.time()
    base_path = Path("C:\\Users\\Robert\\CodingProjects\\combo-tree\\lib\\test\\test replays")
    matcher = ParallelTransitionStateIndexer(base_path)

    replay_files = list(base_path.glob("*.slp"))
    print(f"Found {len(replay_files)} replay files")

    matcher.build_indices(replay_files)
    print("--- %s seconds ---" % (time.time() - start_time))
