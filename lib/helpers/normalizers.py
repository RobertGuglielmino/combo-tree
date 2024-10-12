
from dataclasses import dataclass
import string
from typing import List, Optional, Tuple
import numpy as np
from lib.models.stage_data import STAGE_DATA, StageData

def _normalize_position(position: Tuple[float, float], 
                        stage_id: int,
                        reference_position: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Normalize a position relative to stage boundaries and optionally another position.
    
    Args:
        position: (x, y) coordinates to normalize
        stage_id: ID of the current stage
        reference_position: Optional reference position (e.g., opponent's position)
    
    Returns:
        Normalized position vector with additional features
    """
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
        # Off-stage left
        return -1 - _normalize_distance(stage.left_edge - x, 
                                            stage.left_edge - stage.left_blast)
    elif x > stage.right_edge:
        # Off-stage right
        return 1 + _normalize_distance(x - stage.right_edge,
                                            stage.right_blast - stage.right_edge)
    else:
        # On-stage
        return x / stage.right_edge  # Will be between -1 and 1

def _normalize_y_position(y: float, stage: StageData) -> float:
    """Normalize y position relative to stage height and blast zones"""
    if y < stage.main_platform:
        # Below stage
        return -_normalize_distance(stage.main_platform - y,
                                        stage.main_platform - stage.bottom_blast)
    elif stage.top_platform:
        if y > stage.top_platform:
            # Above top platform
            return _normalize_distance(y - stage.top_platform,
                                        stage.top_blast - stage.top_platform)
        else:
            # Between main stage and top platform
            return y / stage.top_platform  # Will be between 0 and 1
    else:
        return _normalize_distance(y - stage.main_platform,
                                        stage.bottom_blast - stage.main_platform)

def _distance_to_nearest_edge(x: float, y: float, stage: StageData) -> float:
    """Calculate normalized distance to nearest stage edge"""
    if y < stage.main_platform:
        return 1.0  # Below stage
    
    left_dist = abs(x - stage.left_edge)
    right_dist = abs(x - stage.right_edge)
    return min(left_dist, right_dist) / (stage.right_edge - stage.left_edge)

def _distance_to_nearest_platform(x: float, y: float, stage: StageData) -> float:
    """Calculate normalized distance to nearest platform"""
    if not stage.platforms:
        return 1.0
        
    min_dist = float('inf')
    for plat_x1, plat_x2, plat_y in stage.platforms:
        if plat_x1 <= x <= plat_x2:
            # Directly above/below platform
            dist = abs(y - plat_y)
        else:
            # Calculate distance to nearest platform edge
            dist_1 = np.sqrt((x - plat_x1)**2 + (y - plat_y)**2)
            dist_2 = np.sqrt((x - plat_x2)**2 + (y - plat_y)**2)
            dist = min(dist_1, dist_2)
        min_dist = min(min_dist, dist)
        
    return _normalize_distance(min_dist, stage.top_blast - stage.bottom_blast)

def _distance_to_nearest_blast(x: float, y: float, stage: StageData) -> float:
    """Calculate normalized distance to nearest blast zone"""
    left_dist = abs(x - stage.left_blast)
    right_dist = abs(x - stage.right_blast)
    top_dist = abs(y - stage.top_blast)
    bottom_dist = abs(y - stage.bottom_blast)
    
    min_dist = min(left_dist, right_dist, top_dist, bottom_dist)
    max_dist = min(
        stage.right_blast - stage.left_blast,
        stage.top_blast - stage.bottom_blast
    ) / 2
    
    return _normalize_distance(min_dist, max_dist)

def _distance_to_ledge(x: float, y: float, stage: StageData, is_left: bool) -> float:
    """Calculate normalized distance to a specific ledge"""
    ledge_x = stage.left_edge if is_left else stage.right_edge
    ledge_y = stage.main_platform
    
    dist = np.sqrt((x - ledge_x)**2 + (y - ledge_y)**2)
    return _normalize_distance(dist, stage.right_blast - stage.left_blast)

@staticmethod
def _normalize_distance(dist: float, max_dist: float) -> float:
    """Normalize a distance value between 0 and 1"""
    return min(1.0, dist / max_dist)
