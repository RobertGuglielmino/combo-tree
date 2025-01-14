
from typing import Optional, Tuple

import numpy as np

from lib.helpers.normalizers.normalizers import _distance_to_ledge, _distance_to_nearest_blast, _distance_to_nearest_edge, _distance_to_nearest_platform, _normalize_distance
from lib.models.stage_data import STAGE_DATA, StageData


def _normalize_position(position: Tuple[float, float], 
                        stage_id: int,
                        reference_position: Optional[Tuple[float, float]] = None) -> np.ndarray:

    x, y = position
    stage = STAGE_DATA[stage_id]
    
    # 1. Normalize x position relative to stage boundaries (-1 to 1)
    norm_x = _normalize_x_position(x, stage)
    
    # 2. Normalize y position relative to blast zones and platforms
    norm_y = _normalize_y_position(y, stage)
    
    # 3. Calculate distance to key stage features
    edge_dist = _distance_to_nearest_edge(x, y, stage)
    platform_dist = _distance_to_nearest_platform(x, y, stage)
    blast_dist = _distance_to_nearest_blast(x, y, stage)
    
    # 4. Calculate ledge proximity features
    left_ledge_dist = _distance_to_ledge(x, y, stage, is_left=True)
    right_ledge_dist = _distance_to_ledge(x, y, stage, is_left=False)
    
    # 5. If reference position provided, calculate relative features
    if reference_position:
        ref_x, ref_y = reference_position
        dx = _normalize_distance(x - ref_x, stage.right_blast - stage.left_blast)
        dy = _normalize_distance(y - ref_y, stage.top_blast - stage.bottom_blast)
        dist = np.sqrt(dx*dx + dy*dy)
    else:
        dx, dy, dist = 0, 0, 0
        
    return np.array([
        norm_x,                 # Normalized x (-1 to 1)
        norm_y,                 # Normalized y (-1 to 1)
        edge_dist,             # Distance to nearest edge (normalized)
        platform_dist,         # Distance to nearest platform (normalized)
        blast_dist,            # Distance to nearest blast zone (normalized)
        left_ledge_dist,       # Distance to left ledge (normalized)
        right_ledge_dist,      # Distance to right ledge (normalized)
        dx,                    # Relative x distance to reference (if provided)
        dy,                    # Relative y distance to reference (if provided)
        dist                   # Total distance to reference (if provided)
    ])

def _normalize_x_position(x: float, stage: StageData) -> float:
    """Normalize x position relative to stage boundaries"""
    if x < stage.left_edge:
        return -1 - _normalize_distance(stage.left_edge - x, stage.left_edge - stage.left_blast)
    elif stage.right_edge < x:
        return 1 + _normalize_distance(x - stage.right_edge, stage.right_blast - stage.right_edge)
    else:
        return x / stage.right_edge  # Will be between -1 and 1

def _normalize_y_position(y: float, stage: StageData) -> float:
    """Normalize y position relative to stage height and blast zones"""
    if y < stage.main_platform:
        # Below stage
        return -_normalize_distance(stage.main_platform - y, stage.main_platform - stage.bottom_blast)
    elif stage.top_platform:
        if y > stage.top_platform:
            # Above top platform
            return _normalize_distance(y - stage.top_platform, stage.top_blast - stage.top_platform)
        else:
            # Between main stage and top platform
            return y / stage.top_platform  # Will be between 0 and 1
    else:
        return _normalize_distance(y - stage.main_platform, stage.bottom_blast - stage.main_platform)
