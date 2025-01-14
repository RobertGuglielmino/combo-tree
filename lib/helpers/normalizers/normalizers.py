import numpy as np
from lib.models.stage_data import StageData

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
