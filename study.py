import time
from typing import List, Dict
from pathlib import Path
from state_indexer import ParallelTransitionStateIndexer
    
def study():
    start_time = time.time()
    base_path = Path("C:\\Users\\robert\\CodingProjects\\combo-tree\\lib\\test\\test replays")
    matcher = ParallelTransitionStateIndexer(base_path)

    replay_files = list(base_path.glob("*.slp"))
    print(f"Found {len(replay_files)} replay files")

    matcher.build_indices(replay_files)
    print("--- %s seconds ---" % (time.time() - start_time))
