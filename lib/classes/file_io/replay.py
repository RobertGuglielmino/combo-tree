
from lib.models.action_states import ACTION_STATES
from lib.models.characters import Character

class Replay:
    def __init__(self, game_file, main_character: Character = None):
        self.game = game_file
        self.hero_port = None
        self.hero_character = main_character
        self.villain_port = None
        self.villain_character = None

        if main_character:
            for port, player in enumerate(self.game.start.players):
                if player.character == main_character.value:
                    self.hero_port = port
                    self.villain_port = abs(port - 1)
                    self.villain_character = self.game.start.players[self.villain_port]
            if not self.villain_character:
                raise Exception(f"Character put into Replay Class is not in the file: {self.game}. Please remove to default, or add a different character.")
        else:
            self.hero_port = 0
            self.hero_character = self.game.start.players[0]
            self.villain_port = 1
            self.villain_character = self.game.start.players[1]

            
        self.ditto = self.hero_character == self.villain_character



    def stage(self):
        return self.game.start.stage

    def game_length(self):
        return self.game.metadata["lastFrame"]

    def characters(self):
        return self.game.start.players[0], self.game.start.players[1]

    def ports(self):
        return self.hero_port, self.villain_port

    def is_ditto(self):
        return self.ditto

    def hero_state_index(self, frame):
        return self.game.frames.ports[self.hero_port].leader.pre.state[frame].as_py()
    
    def villain_state_index(self, frame):
        return self.game.frames.ports[self.villain_port].leader.pre.state[frame].as_py()

    def hero_state(self, frame):
        return ACTION_STATES[self.game.frames.ports[self.hero_port].leader.pre.state[frame].as_py()]
    
    def villain_state(self, frame):
        return ACTION_STATES[self.game.frames.ports[self.villain_port].leader.pre.state[frame].as_py()]

    def hero_x_position(self, frame):
        return float(self.game.frames.ports[self.hero_port].leader.pre.position.x[frame].as_py())
    
    def hero_y_position(self, frame):
        return float(self.game.frames.ports[self.hero_port].leader.pre.position.y[frame].as_py())
      
    def hero_direction(self, frame):
        return float(self.game.frames.ports[self.hero_port].leader.post.direction[frame].as_py())
    
    def villain_x_position(self, frame):
        return float(self.game.frames.ports[self.villain_port].leader.pre.position.x[frame].as_py())
    
    def villain_y_position(self, frame):
        return float(self.game.frames.ports[self.villain_port].leader.pre.position.y[frame].as_py())
    
    def villain_direction(self, frame):
        return float(self.game.frames.ports[self.villain_port].leader.post.direction[frame].as_py())
    
    def hero_damage(self, frame):
        return self.game.frames.ports[self.hero_port].leader.post.percent[frame].as_py()
    
    def villain_damage(self, frame):
        return self.game.frames.ports[self.villain_port].leader.post.percent[frame].as_py()
    