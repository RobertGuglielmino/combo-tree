

from dataclasses import dataclass
from typing import List
from pathlib import Path
from peppi_py import read_slippi
import json

from lib.classes.file_io.replay import Replay
from lib.models.characters import Character
from lib.models.end_cg_states import END_CHAINGRAB_STATES

@dataclass 
class ChainGrabSequence:
   original_file: str
   start_frame: int
   start_action: str
   end_frame: int
   end_action: str

def save_chaingrab_sequences(sequences: List[ChainGrabSequence], output_path: str):
    """Save sequences as JSON mapping to original files and frames"""
    file_names = []
    data = []

    for seq in sequences:
        if seq.original_file not in file_names:
            file_names.append(seq.original_file)

    print(file_names)
    
    for file in file_names:
        new_list = [{
            "start_time_game_clock": f"{(7 - int(n.start_frame / 3600)):02}:{59 - (int(n.start_frame / 60) % 60):02}.{99 - int((n.start_frame % 60) * 100 / 60):02}",
            "start_time_player": f"{int(n.start_frame / 3600)}:{(int(n.start_frame / 60) % 60):02}",
            "start_frame": n.start_frame,
            "end_frame": n.end_frame,
            "length": '%.2f'%((n.end_frame - n.start_frame) / 60)
        } for n in sequences if n.original_file == file]
        data.append({
            "file": file,
            "chaingrabs": new_list
        })

    with open(output_path, 'w') as f:
        json.dump(data, f)

def extract_chaingrabs(replay_path: str, log_path: str = None) -> List[ChainGrabSequence]:
    game = Replay(read_slippi(replay_path), Character.MARTH)

    if game.get_stage() != 32:
        return []
        
    if log_path:
        log_character_states(game, log_path)

    sequences = []
    current_sequence = None

    for frame in range(game.get_game_length()):
        if game.get_hero_state_at_frame(frame) == "CATCH_PULL" and current_sequence is None:
            current_sequence = ChainGrabSequence(
                original_file=replay_path,
                start_frame=frame,
                start_action="",
                end_frame=None,
                end_action=""
            )
        elif current_sequence is not None:
            if game.get_villain_state_at_frame(frame) in END_CHAINGRAB_STATES:
                current_sequence.end_frame = frame
                sequences.append(current_sequence)
                current_sequence = None

    return sequences

def log_character_states(game: Replay, output_file: str):
    """Log frame-by-frame states for both characters"""
    with open(output_file, 'w') as f:
        f.write("Frame, , Marth State, , Fox State\n")
        for frame in range(game.get_game_length()):
            f.write(f"{frame}, \
                    {game.get_hero_state_at_frame_index(frame)}, \
                    {game.get_hero_state_at_frame(frame)}, \
                    {game.get_villain_state_at_frame_index(frame)}, \
                    {game.get_villain_state_at_frame(frame)}\n")

def main():
    all_sequences = []
    replay_path = Path("E:\SlippiDataSet")
    output_path = "C:\\Users\\robert\\CodingProjects\\combo-tree\\lib\\test\\test_chaingrab_replays\\chaingrabs_data.json"

    for file in replay_path.glob("*.slp"):
        log_file = str(file).replace('.slp', '_states.csv')
        sequences = extract_chaingrabs(str(file), log_file)
        all_sequences.extend(sequences)
        
    save_chaingrab_sequences(all_sequences, output_path)

if __name__ == "__main__":
    main()