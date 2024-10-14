from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import pyarrow as pa

from lib.helpers.matchup_key import MatchupKey
from lib.helpers.normalizers import _normalize_position
from lib.models.action_states import ACTION_STATES
from lib.models.characters_by_id import CHARACTERS_BY_ID
from lib.models.intermediary_states import INTERMEDIARY_STATES

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
    p1_stocks: int
    
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
    p2_stocks: int


@dataclass
class ActionSequence:
    initial_state: int
    intermediary_states: List[int]
    final_action: int

ACTION_STATE_COUNT = len(ACTION_STATES)

def _get_matchup_key(game) -> MatchupKey:
    active_ports = list(filter(lambda x: x, game.start.players))
    p1_char = CHARACTERS_BY_ID[active_ports[0].character]
    p2_char = CHARACTERS_BY_ID[active_ports[1].character]
    return MatchupKey(p1_char, p2_char)

def _frame_to_game_state(frame, game_start, index: int, is_p1_perspective: bool) -> GameState:
    p1_data = frame.ports[0].leader
    p2_data = frame.ports[1].leader

    p1_pre = p1_data.pre
    p1_post = p1_data.post
    p2_pre = p2_data.pre
    p2_post = p2_data.post

    return GameState(
        frame_index=_to_int(index),
        stage_id=_to_int(game_start.stage),
        
        # Player 1 state
        p1_position=(_to_float(p1_pre.position.x[index]), _to_float(p1_pre.position.y[index])),
        p1_velocity=(_to_float(p1_post.position.x[index]) - _to_float(p1_pre.position.x[index]), _to_float(p1_post.position.y[index]) - _to_float(p1_pre.position.y[index])),
        p1_damage=_to_float(p1_post.percent[index]),
        p1_state=_to_int(p1_post.state[index]),
        p1_direction=_to_float(p1_post.direction[index]),
        p1_character=_to_int(game_start.players[0].character),
        p1_inputs=(p1_pre.buttons[index]),
        p1_jumps_remaining=_to_int(p1_post.jumps[index]),
        p1_on_ground=_to_int(p1_post.airborne[index]),
        p1_shield_strength=_to_float(p1_post.shield[index]),
        p1_stocks=_to_int(p1_post.stocks[index]),
        
        # Player 2 state
        p2_position=(_to_float(p2_pre.position.x[index]), _to_float(p2_pre.position.y[index])),
        p2_velocity=(_to_float(p2_post.position.x[index]) - _to_float(p2_pre.position.x[index]), _to_float(p2_post.position.y[index]) - _to_float(p2_pre.position.y[index])),
        p2_damage=_to_float(p2_post.percent[index]),
        p2_state=_to_int(p2_post.state[index]),
        p2_direction=_to_float(p2_post.direction[index]),
        p2_character=_to_int(game_start.players[0].character),
        p2_jumps_remaining=_to_int(p2_post.jumps[index]),
        p2_on_ground=_to_int(p2_post.airborne[index]),
        p2_shield_strength=_to_float(p2_post.shield[index]),
        p2_stocks=_to_int(p2_post.stocks[index]),
    )

def _game_state_to_vector(state: GameState) -> np.ndarray:
    p1_pos = _normalize_position(state.p1_position, state.stage_id)
    p2_pos = _normalize_position(state.p2_position, state.stage_id)
    
    vector_components = [
        *p1_pos,
        *p2_pos,
        *(p1_pos - p2_pos),
        state.p1_damage / 999.0,
        state.p2_damage / 999.0,
        *_one_hot_encode(state.p1_state, num_actions=ACTION_STATE_COUNT),
        *_one_hot_encode(state.p2_state, num_actions=ACTION_STATE_COUNT),
        # state.p1_inputs,
        np.sin(state.p1_direction * np.pi),
        np.cos(state.p1_direction * np.pi),
        np.sin(state.p2_direction * np.pi),
        np.cos(state.p2_direction * np.pi)
        # Add more relevant features as needed
    ]

    return np.asarray(vector_components, dtype="object").flatten()


def _get_next_significant_action(game, start_index: int, is_p1_perspective: bool) -> ActionSequence:

    initial_state = game.frames.ports[0].leader.post.state[start_index]
    intermediary_states = []
    
    for i in range(start_index + 1, game.metadata["lastFrame"]):
        current_state = game.frames.ports[0].leader.post.state[i]
        if ACTION_STATES[current_state.as_py()] in INTERMEDIARY_STATES:
            intermediary_states.append(current_state)
        else:
            return ActionSequence(initial_state, intermediary_states, current_state)
    
    # If we reach the end without finding a significant action
    return ActionSequence(initial_state, intermediary_states, "END_OF_REPLAY")

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

def print_action_sequence(sequence: ActionSequence) -> str:
    action_list = []
    action_list.append(ACTION_STATES[sequence.initial_state.as_py()])
    [action_list.append(ACTION_STATES[state.as_py()]) for state in sequence.intermediary_states]
    action_list.append(ACTION_STATES[sequence.final_action.as_py()])

    return "->".join(action_list)


# helper functions for encoding
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

def _process_inputs(pre_state) -> Dict[str, any]:
    """Process raw inputs into a standardized format"""
    return {
        'main_stick': (pre_state.joystick.x, pre_state.joystick.y),
        'c_stick': (pre_state.cstick.x, pre_state.cstick.y),
        'l_trigger': pre_state.triggers.physical.l,
        'r_trigger': pre_state.triggers.physical.r,
        'buttons': pre_state.buttons.physical.pressed()
    }

def _to_float(value):
    """Convert PyArrow scalar to float."""
    if isinstance(value, pa.Scalar):
        return float(value.as_py())
    return float(value)

def _to_int(value):
    """Convert PyArrow scalar to int."""
    if isinstance(value, pa.Scalar):
        return int(value.as_py())
    return int(value)
