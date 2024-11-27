from typing import Dict, List, Tuple
import numpy as np
import psutil
from game_states import _frame_to_vector, _get_next_significant_action
from lib.helpers.matchup_key import MatchupKey, _get_matchup_key
from peppi_py import read_slippi

# input:   a replay file
# output:  a dictionary of the replays Character Matchup (fox vs. marth) -> all the frames of the replay as normalized vectors
def process_replay(replay_file) -> Dict[MatchupKey, List[Tuple[np.ndarray, str]]]:
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

            # Process from P1's perspective
            results[matchup] = _process_replay_perspective(game, is_p1_perspective=True)

            # If it's not a mirror match, process from P2's perspective
            if not matchup.is_mirror:
                reversed_matchup = matchup.reversed
                results[reversed_matchup] = _process_replay_perspective(game, is_p1_perspective=False)

            # with open(cache_file, 'wb') as f:
            #     pickle.dump(results, f)
            return results
        
        except Exception as e:
            print(f"Error reading and processing slippi file {replay_file}: {e}")
            return {}


def process_replays_in_chunks(replay_files, chunk_size=5, perspective=True):
    for i in range(0, len(replay_files), chunk_size):
        chunk = replay_files[i:i + chunk_size]
        chunk_data = []
        print ("\033[A                                                                                          \033[A")
        print(f"\rProcessing replays {i+1} to {min(i+chunk_size, len(replay_files))} of {len(replay_files)} ---- Current RAM memory % usage: {psutil.virtual_memory()[2]}")
        
        
        for file in chunk:
            try:
                game = read_slippi(str(file))
                print ("\033[A                             \033[A")
                states = _process_replay_perspective_no_sequence(game, game.start.players[0].character == 18)
                chunk_data.extend(states)
                
                del game
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
            
        yield chunk_data
        
        del chunk_data


def _process_replay_perspective(game, is_p1_perspective: bool) -> List[Tuple[np.ndarray, str]]:
    processed_states = []
    i = 0

    while i < game.metadata["lastFrame"]:
        state_vector = _frame_to_vector(game, i, is_p1_perspective)
        action_sequence = _get_next_significant_action(game, i, is_p1_perspective)
        i += len(action_sequence.intermediary_states) + 1


        processed_states.append((state_vector, action_sequence))

    return processed_states

def _process_replay_perspective_no_sequence(game, is_p1_perspective: bool) -> List[np.ndarray]:
    processed_states = []
    i = 0

    while i < game.metadata["lastFrame"]:
        state_vector = _frame_to_vector(game, i, is_p1_perspective)
        action_sequence = _get_next_significant_action(game, i, is_p1_perspective)
        i += len(action_sequence.intermediary_states) + 1


        processed_states.append(state_vector)

    return processed_states