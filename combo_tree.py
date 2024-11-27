# Example usage
import cProfile
from pathlib import Path
from predict import predict
from study import study

def main():
    base_path = Path("C:\\Users\\robert\\CodingProjects\\combo-tree\\lib\\test\\test replays")

    cProfile.run('study()')
    # study(base_path)
    predict(base_path)

if __name__ == "__main__":
    main()