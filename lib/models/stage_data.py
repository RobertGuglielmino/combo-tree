from dataclasses import dataclass
import string
from typing import List, Tuple

@dataclass
class StageData:
    """Contains stage boundaries and platform data"""
    stage_name: string

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


STAGE_DATA = [
    # Values from https://www.ssbwiki.com/Stage_data_(SSBM)
    StageData(
        stage_name="FINAL_DESTINATION",
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
    StageData(
        stage_name="BATTLEFIELD",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    StageData(
        stage_name="YOSHIS_STORY",
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
    )
]
