from dataclasses import dataclass
import time
import numpy as np
from annoy import AnnoyIndex
from typing import List, Dict, Optional, Tuple
import pickle
from slippi import Game
from pathlib import Path
import concurrent.futures
from state_weights import StateWeights
from lib.helpers.matchup_key import MatchupKey
from lib.helpers.normalizers import _normalize_position
from lib.models.action_states import ACTION_STATES
# from stage_data import PositionNormalizer, StageData

def _one_hot_encode(value: int, num_actions: int) -> np.ndarray:
    """
    Create a one-hot encoded vector for a given value.
    
    Args:
        value: Integer value to encode.
        num_actions: Total number of possible classes.
    
    Returns:
        Numpy array with one-hot encoding.
    """
    encoding = np.zeros(num_actions)
    if 0 <= value < num_actions:
        encoding[value] = 1
    return encoding

@dataclass
class StageData:
    """Contains stage boundaries and platform data"""
    state_id: int

    # Main stage
    left_edge: float
    right_edge: float
    top_platform: float
    main_platform: float  # Stage height
    
    # Blast zones
    left_blast: float
    right_blast: float
    top_blast: float
    bottom_blast: float
    
    # Platform positions (if any)
    platforms: List[Tuple[float, float, float]]  # (x1, x2, height) for each platform

@dataclass
class GameState:
    """Represents a single frame's state for pattern matching"""
    frame_index: int
    stage_id: Optional[int]
    
    # Player 1 state
    p1_position: Tuple[float, float]
    p1_velocity: Tuple[float, float]
    p1_damage: float
    p1_state: int  # ActionState enum value
    p1_direction: int  # 1 for right, -1 for left
    p1_character: int  # Character enum value
    p1_inputs: Dict[str, any]  # Processed inputs
    p1_jumps_remaining: int
    p1_on_ground: bool
    p1_shield_strength: Optional[float]
    
    # Player 2 state (opponent)
    p2_position: Tuple[float, float]
    p2_velocity: Tuple[float, float]
    p2_damage: float
    p2_state: int
    p2_direction: int
    p2_character: int
    p2_jumps_remaining: int
    p2_on_ground: bool
    p2_shield_strength: Optional[float]

 

ACTION_STATE_COUNT = 1762
STAGE_DATA = {
    # Values from https://www.ssbwiki.com/Stage_data_(SSBM)
    "Stage.FINAL_DESTINATION": StageData(
        state_id=0,
        left_edge=-85.5607,
        right_edge=85.5607,
        top_platform=0,
        main_platform=0,
        left_blast=-198.75,
        right_blast=198.75,
        top_blast=180.0,
        bottom_blast=-140.0,
        platforms=[]
    ),
    "Stage.BATTLEFIELD": StageData(
        state_id=1,
        left_edge=-68.4,
        right_edge=68.4,
        top_platform=54.4,
        main_platform=0,
        left_blast=-224.0,
        right_blast=224.0,
        top_blast=200.0,
        bottom_blast=-108.8,
        platforms=[
            (-57.6, 57.6, 27.2),  # Side platforms
            (-18.8, 18.8, 54.4)   # Top platform
        ]
    ),
    "Stage.YOSHIS_STORY": StageData(
        state_id=2,
        left_edge=-56.0,
        right_edge=56.0,
        top_platform=41.25,
        main_platform=0,
        left_blast=-175.7,
        right_blast=175.7,
        top_blast=168.0,
        bottom_blast=-91.0,
        platforms=[
            (-56.0, 56.0, 23.45),  # Side platforms
            (-15.65, 15.65, 41.25)  # Top platform
        ]
    ),
    # Add other legal stages...
}


# Additional utility function for button processing
def _encode_buttons(button_state: List[str]) -> np.ndarray:
    """
    Encode button states into a binary array.
    
    Args:
        button_state: Dictionary of button states (True for pressed, False for not pressed).
    
    Returns:
        Numpy array of binary button states.
    """
    button_order = ['a', 'b', 'x', 'y', 'l', 'r', 'z', 'start', 'd_up', 'd_down', 'd_left', 'd_right']
    return np.array([int(button in button_state) for button in button_order])
   
def _frame_to_game_state(frame, game_metadata) -> GameState:
    """Convert a Slippi frame into our GameState representation"""
    # Get player ports (handle potential port swapping)
    p1_port = frame.ports[0].leader
    p2_port = frame.ports[1].leader
    
    # Process pre-frame data for both players
    p1_pre = p1_port.pre
    p2_pre = p2_port.pre
    
    # Process post-frame data (if available) for additional info
    p1_post = p1_port.post if hasattr(p1_port, 'post') else None
    p2_post = p2_port.post if hasattr(p2_port, 'post') else None
    
    # Get character info from metadata
    p1_char = list(game_metadata.players[0].characters.keys())[0].value
    p2_char = list(game_metadata.players[1].characters.keys())[0].value
    
    # Process inputs for player 1
    p1_inputs = _process_inputs(p1_pre)
    
    # Create GameState instance
    return GameState(
        frame_index=frame.index,
        stage_id=frame.start.stage if hasattr(frame.start, 'stage') else "Stage.FINAL_DESTINATION",
        
        # Player 1 state
        p1_position=(p1_pre.position.x, p1_pre.position.y),
        p1_velocity=(p1_post.position.x - p1_pre.position.x if p1_post else 0,
                    p1_post.position.y - p1_pre.position.y if p1_post else 0),
        p1_damage=p1_pre.damage,
        p1_state=p1_pre.state,
        p1_direction=1 if p1_pre.direction.value > 0 else -1,
        p1_character=p1_char,
        p1_inputs=p1_inputs,
        p1_jumps_remaining=p1_post.jumps,
        p1_on_ground=_is_grounded(p1_pre.state),
        p1_shield_strength=p1_pre.shield_strength if hasattr(p1_pre, 'shield_strength') else None,
        
        # Player 2 state
        p2_position=(p2_pre.position.x, p2_pre.position.y),
        p2_velocity=(p2_post.position.x - p2_pre.position.x if p2_post else 0,
                    p2_post.position.y - p2_pre.position.y if p2_post else 0),
        p2_damage=p2_pre.damage,
        p2_state=p2_pre.state,
        p2_direction=1 if p2_pre.direction.value > 0 else -1,
        p2_character=p2_char,
        p2_jumps_remaining=p2_post.jumps,
        p2_on_ground=_is_grounded(p2_pre.state),
        p2_shield_strength=p2_pre.shield_strength if hasattr(p2_pre, 'shield_strength') else None
    )

def _process_inputs(pre_state) -> Dict[str, any]:
    """Process raw inputs into a standardized format"""
    return {
        'main_stick': (pre_state.joystick.x, pre_state.joystick.y),
        'c_stick': (pre_state.cstick.x, pre_state.cstick.y),
        'l_trigger': pre_state.triggers.physical.l,
        'r_trigger': pre_state.triggers.physical.r,
        'buttons': pre_state.buttons.physical.pressed()
    }

def _is_grounded(state) -> bool:
    """Determine if a player is on the ground based on their action state"""
    ground_states = {
        # Add all relevant ground action states
        'STANDING': True,
        'WALK_SLOW': True,
        'WALK_MIDDLE': True,
        'WALK_FAST': True,
        'DASH': True,
        'RUN': True,
        'CROUCH': True,
        'LANDING': True,
        # Add more as needed...
    }
    return ground_states.get(str(state), False)

def _get_next_action(frame) -> str:
    """Determine the next action taken from a frame"""
    return frame.ports[0].leader.post.state
    # p1_post = frame.ports[0].leader.post
    
    # Define a hierarchy of actions to check
    # if p1_post.state:
    #     return f"ATTACK_{str(p1_post.state)}"
    # elif _is_defensive_action(p1_post.state):
    #     return f"DEFENSE_{str(p1_post.state)}"
    # elif _is_movement_action(p1_post.state):
    #     return f"MOVEMENT_{str(p1_post.state)}"
    # else:
    #     return p1_post.state

def _is_defensive_action(state) -> bool:
    """Check if an action state is defensive"""
    defensive_states = {
        'SHIELD_START',
        'SHIELD',
        'SHIELD_RELEASE',
        'SPOT_DODGE',
        'ROLL_FORWARD',
        'ROLL_BACKWARD',
        # Add more as needed...
    }
    return str(state) in defensive_states

def _is_movement_action(state) -> bool:
    """Check if an action state is movement-related"""
    movement_states = {
        'DASH',
        'RUN',
        'JUMP',
        'DOUBLE_JUMP',
        'AIR_DODGE',
        'WAVEDASH',
        # Add more as needed...
    }
    return str(state) in movement_states

class MatchupSpecificMatcher:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.vector_dim = 3578  # Adjust based on your state vector size
        self.weights = StateWeights()
        self.matchup_indices: Dict[MatchupKey, AnnoyIndex] = {}
        self.matchup_caches: Dict[MatchupKey, Dict[int, str]] = {} 
        self.stage_data = STAGE_DATA
    
    def _get_matchup_keys(self, game) -> List[MatchupKey]:
        """Extract matchup information from game"""
        active_ports = list(filter(lambda x: x, game.metadata.players))
        p1_char = list(active_ports[0].characters.keys())[0].name
        p2_char = list(active_ports[1].characters.keys())[0].name
        return [MatchupKey(p1_char, p2_char), MatchupKey(p2_char, p1_char)]
    
    def _create_vector_components(self, state: GameState) -> Dict[str, np.ndarray]:
        """Create dictionary of vector components before applying weights"""
        p1_pos = np.array(_normalize_position(state.p1_position, state.stage_id))
        p2_pos = np.array(_normalize_position(state.p2_position, state.stage_id))
        
        # Calculate relative position (vector from p1 to p2)
        relative_pos = p2_pos - p1_pos
        
        return {
            'p1_pos': p1_pos,
            'p2_pos': p2_pos,
            'relative_pos': relative_pos,
            'p1_damage': np.array([state.p1_damage / 999.0]),
            'p2_damage': np.array([state.p2_damage / 999.0]),
            'p1_state': _one_hot_encode(state.p1_state, num_actions=ACTION_STATE_COUNT),
            'p2_state': _one_hot_encode(state.p2_state, num_actions=ACTION_STATE_COUNT),
            'inputs': {
                'main_stick': np.array(state.p1_inputs['main_stick']),
                'c_stick': np.array(state.p1_inputs['c_stick']),
                'triggers': np.array([state.p1_inputs['l_trigger'], state.p1_inputs['r_trigger']]),
                'buttons': np.array(_encode_buttons(state.p1_inputs['buttons']))
            },
            'p1_direction': np.array([
                np.sin(state.p1_direction * np.pi),
                np.cos(state.p1_direction * np.pi)
            ]),
            'p2_direction': np.array([
                np.sin(state.p2_direction * np.pi),
                np.cos(state.p2_direction * np.pi)
            ])
        }
    
    def process_replay(self, pre_game) -> None:
        """Process a replay file and add to appropriate matchup index"""
        game = Game(pre_game)
        matchups = self._get_matchup_keys(game)

        processed_states = []
        
        for i in range(len(game.frames) - 1):
            current_frame = game.frames[i]
            print(current_frame)
            next_frame = game.frames[i + 1]
            
            state = _frame_to_game_state(current_frame, game.metadata)
            vector_components = self._create_vector_components(state)
            state_vector = self.weights.apply_weights(vector_components)
            
            next_action = _get_next_action(next_frame)
            
            processed_states.append((state_vector, next_action))
            
        return [{matchups[0]: processed_states}, {matchups[1]: processed_states}]
            
    
    def build_indices(self, replay_files: List[Path], n_trees: int = 10):
        """Process multiple replay files and build indices for each matchup"""
        # Process replays in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(self.process_replay, replay_file): replay_file for replay_file in replay_files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            replay_file = future_to_file[future]
            print("future_to_file")
            print(replay_file)
            try:
                result = future.result()
                self._update_indices(result)
            except Exception as exc:
                print(f'{replay_file} generated an exception: {exc}')

        print(self.matchup_indices)

        # Build indices for each matchup
        for matchup, index in self.matchup_indices.items():
            print(f"Building index for matchup: {matchup}")
            index.build(n_trees)

        self.save_indices()
    
    def find_similar_states(self, 
                          current_state: GameState,
                          matchup: MatchupKey,
                          n_neighbors: int = 10) -> List[Tuple[str, float]]:
        """Find similar states for a specific matchup"""
        if matchup not in self.matchup_indices:
            raise ValueError(f"No data available for matchup: {matchup}")
            
        index = self.matchup_indices[matchup]
        cache = self.matchup_caches[matchup]
        
        vector_components = self._create_vector_components(current_state)
        state_vector = self.weights.apply_weights(vector_components)
        
        indices, distances = index.get_nns_by_vector(
            state_vector, n_neighbors, include_distances=True
        )
        
        return [(ACTION_STATES[cache[idx]], dist) for idx, dist in zip(indices, distances)]
    
    def save_indices(self):
        """Save all matchup indices and caches"""
        print("self.matchup_indice   222s")
        print(self.matchup_indices)
        for matchup in self.matchup_indices:
            print(matchup)
            matchup_path = self.base_path / str(matchup)
            matchup_path.mkdir(parents=True, exist_ok=True)
            
            self.matchup_indices[matchup].save(str(matchup_path / "index.ann"))
            with open(matchup_path / "cache.pkl", 'wb') as f:
                pickle.dump(self.matchup_caches[matchup], f)
   
    def _update_indices(self, result: Dict[MatchupKey, List[Tuple[np.ndarray, str]]]):
        for matchup, states in result.items():
            if matchup not in self.matchup_indices:
                self.matchup_indices[matchup] = AnnoyIndex(self.vector_dim, 'euclidean')
                self.matchup_caches[matchup] = {}
            
            index = self.matchup_indices[matchup]
            cache = self.matchup_caches[matchup]
            current_index_size = len(cache)
            
            for i, (state_vector, next_action) in enumerate(states):
                index.add_item(current_index_size + i, state_vector)
                cache[current_index_size + i] = next_action
     
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

# Example usage
def main():
    base_path = Path("C:\\Users\\Robert\\CodingProjects\\combo-tree\\test replays")
    matcher = MatchupSpecificMatcher(base_path)
    start_time = time.time()

    # Build indices from replay files
    replay_files = list(base_path.glob("*.slp"))
    matcher.build_indices(replay_files)


    print("--- %s seconds ---" % (time.time() - start_time))
    
    game = Game('./test user replays/Game_20221028T163330.slp')
    # Later, get recommendations for specific matchup
    # print(game.frames[9880])
    # print(list(filter(lambda x: x["index"] == 9880, game.frames)))
    current_state = _frame_to_game_state(game.frames[9880], game.metadata)  # Create from current frame
    matchup = MatchupKey(
        p1_char="FOX",
        p2_char="MARTH"
    )
    
    # matcher.load_indices()
    # similar_states = matcher.find_similar_states(current_state, matchup)
    # print("Recommended actions and their distances:")
    # for action, distance in similar_states:
    #     print(f"Action: {action}, Distance: {distance:.3f}")

if __name__ == "__main__":
    main()