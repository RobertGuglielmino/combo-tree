import os
from pathlib import Path
import pickle
import time
from typing import Dict, List, Tuple
from annoy import AnnoyIndex
import concurrent
import numpy as np

from game_states import ActionSequence, _compress_states
from lib.helpers.matchup_key import MatchupKey
from lib.helpers.process_replay import process_replay
from lib.helpers.update_progress import update_progress


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
            result = process_replay(replay_file)
            if result:
                return result
        except Exception as e:
            print(f"Error processing replay file: {replay_file}: {e}")
        return None

    def build_indices(self, replay_files: List[str], n_trees: int = 10):
        all_results = {}
        total_files = len(replay_files)
        start_time = time.time()
        
        print(f"Starting to process {total_files} replay files...")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process files with simple progress indicator
            results = []
            for i, result in enumerate(executor.map(self.process_and_update, replay_files)):
                if i % 10 == 0 or i == total_files - 1:  # Update more frequently
                    update_progress(i + 1, total_files, start_time)
                results.append(result)

        print("Processing complete. Aggregating results...")

        self._update_indices(all_results)

        for index in self.matchup_indices.values():
            index.build(n_trees)

        self.save_indices()


    def _update_indices(self, results: Dict[MatchupKey, List[Tuple[np.ndarray, ActionSequence]]]):
        print("Beginning index creation")
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
    