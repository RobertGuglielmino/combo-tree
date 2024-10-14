from typing import NamedTuple

class MatchupKey(NamedTuple):
    p1_char: str  # Character enum value
    p2_char: str
    
    def __str__(self):
        return f"{self.p1_char}_vs_{self.p2_char}"

    @property
    def reversed(self):
        return MatchupKey(self.p2_char, self.p1_char)

    @property
    def is_mirror(self):
        return self.p1_char == self.p2_char