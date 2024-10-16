from typing import NamedTuple

from lib.models.characters_by_id import CHARACTERS_BY_ID

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
    

def _get_matchup_key(game) -> MatchupKey:
    active_ports = list(filter(lambda x: x, game.start.players))
    p1_char = CHARACTERS_BY_ID[active_ports[0].character]
    p2_char = CHARACTERS_BY_ID[active_ports[1].character]
    return MatchupKey(p1_char, p2_char)
