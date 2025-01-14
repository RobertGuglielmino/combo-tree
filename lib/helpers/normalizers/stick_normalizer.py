import numpy as np
from typing import Tuple, List

def _normalize_stick_position(stick_position: Tuple[float, float]) -> np.ndarray:
    """
    Normalize analog stick inputs to a unit circle.
    
    Args:
        stick_position: Tuple of (x, y) coordinates from -1 to 1.
    
    Returns:
        Numpy array of normalized (x, y) coordinates.
    """
    x, y = stick_position
    magnitude = np.sqrt(x**2 + y**2)
    
    if magnitude == 0:
        return np.array([0.0, 0.0])
    
    # Normalize to unit circle
    normalized_x = x / magnitude
    normalized_y = y / magnitude
    
    # Apply deadzone (optional)
    deadzone = 0.2875  # Melee's approximate deadzone
    if magnitude < deadzone:
        return np.array([0.0, 0.0])
    
    # Rescale the normalized position to account for deadzone
    rescaled_magnitude = (magnitude - deadzone) / (1 - deadzone)
    return np.array([normalized_x * rescaled_magnitude, normalized_y * rescaled_magnitude])