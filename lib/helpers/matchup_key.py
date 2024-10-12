from typing import NamedTuple

class MatchupKey(NamedTuple):
    p1_char: str  # Character enum value
    p2_char: str
    
    def __str__(self):
        return f"{self.p1_char}_vs_{self.p2_char}"