from dataclasses import dataclass
from typing import Any, Dict, List,  Optional, Tuple
import numpy as np
import pyarrow as pa

from lib.classes.file_io.replay import Replay
from lib.helpers.normalizers.character_normalizers import _normalize_position
from lib.models.action_states import ACTION_STATES
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

# takes a single frame of the game, and turns it into a flattened, normalized array, to be run through nearest neighbor search
def _frame_to_vector(game: Replay, index: int, port: int) -> np.ndarray:
    villain_port = abs(port-1)
    ACTION_STATE_COUNT = len(ACTION_STATES)
    stage_id=game.stage()

    # normalize position is biggest bottleneck
    p1_pos = _normalize_position((game.hero_x_position(index), game.hero_y_position(index)), stage_id)
    p2_pos = _normalize_position((game.villain_x_position(index), game.villain_y_position(index)), stage_id)
    p1_direction = game.hero_direction(index)
    p2_direction = game.villain_direction(index)

    frame_index=_to_int(index)

    # to add
    # p1_velocity=(_to_float(p1_post.position.x[index]) - _to_float(p1_pre.position.x[index]), _to_float(p1_post.position.y[index]) - _to_float(p1_pre.position.y[index])),
    #     p1_inputs=(p1_pre.buttons[index]),
    #     p1_jumps_remaining=_to_int(p1_post.jumps[index]),
    #     p1_on_ground=_to_int(p1_post.airborne[index]),
    #     p1_shield_strength=_to_float(p1_post.shield[index]),
    #     p1_stocks=_to_int(p1_post.stocks[index]),

    vector_components = [
        # *p1_pos,
        # *p2_pos,
        # *(p1_pos - p2_pos),
        # np.array([_to_float(game.hero_damage(index)) / 300.0])[port],
        np.array([_to_float(game.villain_damage(index)) / 300.0])[port],
        *_one_hot_encode(game.hero_state_index(index), num_actions=ACTION_STATE_COUNT),
        *_one_hot_encode(game.villain_state_index(index), num_actions=ACTION_STATE_COUNT),
        # state.p1_inputs,
        # np.sin(p1_direction * np.pi),
        # np.cos(p1_direction * np.pi),
        # np.sin(p2_direction * np.pi),
        # np.cos(p2_direction * np.pi)
        # Add more relevant features as needed
    ]

    return np.asarray(vector_components, dtype="object").flatten()

# some frames of melee (jumpsquat, grab pull, landing lag)
# are not a choice that the player has made, so we will ignore them until
# we find an action from a new button that the player has pressed
def _get_next_significant_action(game: Replay, start_index: int, port: int) -> ActionSequence:
    villain_port = abs(port-1)

    initial_state = game.hero_state_index(start_index)
    intermediary_states = []

    
    for i in range(start_index + 1, game.game_length()):
        current_state = game.hero_state_index(i)
        if ACTION_STATES[current_state] in INTERMEDIARY_STATES:
            intermediary_states.append(current_state)
        else:
            return ActionSequence(initial_state, intermediary_states, current_state)
    
    # If we reach the end without finding a significant action
    return ActionSequence(initial_state, intermediary_states, "END_OF_REPLAY")

def _compress_states(states: List[Tuple[np.ndarray, ActionSequence]]) -> List[Tuple[np.ndarray, ActionSequence]]:
    compressed_states = []
    current_action = None
    action_states = []

    for state_vector, action_sequence in states:
        if action_sequence.final_action != current_action:
            if current_action is not None:
                avg_state_vector = np.mean([s for s, _ in action_states], axis=0)
                compressed_states.append((avg_state_vector, action_states[0][1]))
            current_action = action_sequence.final_action
            action_states = []
        action_states.append((state_vector, action_sequence))

    if action_states:
        avg_state_vector = np.mean([s for s, _ in action_states], axis=0)
        compressed_states.append((avg_state_vector, action_states[0][1]))

    return compressed_states


# helper functions for encoding
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

def _process_inputs(pre_state) -> Dict[str, any]:
    """Process raw inputs into a standardized format"""
    return {
        'main_stick': (pre_state.joystick.x, pre_state.joystick.y),
        'c_stick': (pre_state.cstick.x, pre_state.cstick.y),
        'l_trigger': pre_state.triggers.physical.l,
        'r_trigger': pre_state.triggers.physical.r,
        'buttons': pre_state.buttons.physical.pressed()
    }
    
def _one_hot_encode(value: int, num_actions: int) -> np.ndarray:
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
