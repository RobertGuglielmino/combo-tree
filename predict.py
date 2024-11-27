from typing import List, Dict, Tuple
from pathlib import Path
from annoy import AnnoyIndex
from peppi_py import read_slippi
import pickle
from game_states import _frame_to_vector, print_action_sequence
from lib.models.action_states import ACTION_STATES
from lib.helpers.matchup_key import MatchupKey, _get_matchup_key

class TransitionStatePredictor:
    def __init__(self, base_path: Path, vector_dim: int = 802):
        self.base_path = base_path
        self.vector_dim = vector_dim
        self.matchup_indices: Dict[MatchupKey, AnnoyIndex] = {}
        self.matchup_caches: Dict[MatchupKey, Dict[int, str]] = {}

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

    def find_similar_states(self, state_vector, matchup: MatchupKey, n_neighbors: int = 10) -> List[Tuple[str, float]]:

        if matchup not in self.matchup_indices:
            raise ValueError(f"No data available for matchup: {matchup}")

        index = self.matchup_indices[matchup]
        cache = self.matchup_caches[matchup]

        indices, distances = index.get_nns_by_vector(state_vector, n_neighbors, include_distances=True)

        return [(cache[idx], dist) for idx, dist in zip(indices, distances)]
    
def predict(base_path):
    matcher = TransitionStatePredictor(base_path)
    matcher.load_indices()
    test_game = read_slippi('./Game_20220910T180650.slp')
    test_frame = 460
    print(ACTION_STATES[test_game.frames.ports[1].leader.pre.state[test_frame].as_py()])

    # # Example of using the matcher
    similar_states = matcher.find_similar_states(_frame_to_vector(test_game, test_frame, True), _get_matchup_key(test_game))
    print("Similar states and their next actions:")
    for next_action, distance in similar_states:
        print(f"Next action: {print_action_sequence(next_action)}, Distance: {distance:.3f}")
