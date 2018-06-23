import torch
from basic_game import Game, Direction


def dir2tensor(direction: Direction):
    # noinspection PyTypeChecker
    direction_list = [0.] * len(Direction)
    direction_list[direction.value] = 1.
    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    return torch.tensor([direction_list], dtype=torch.float32)


def game2tensor(game: Game):

    if game is None:
        return None
    elif isinstance(game, bool):
        return None

    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    return torch.tensor([(game.food, *game.snake)], dtype=torch.float32)
