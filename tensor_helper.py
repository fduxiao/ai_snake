import torch
from basic_game import Game, Direction, MapStatus


def dir2tensor(direction: Direction):
    # noinspection PyTypeChecker
    direction_list = [0.] * len(Direction)
    direction_list[direction.value] = 1.
    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    return torch.tensor([direction_list], dtype=torch.float32)


def game2tensor(game: Game, is_map=False):

    if game is None:
        return None
    elif isinstance(game, bool):
        return None

    game_map = game
    if not is_map:
        game_map = game.draw_map()
    game_list = list()
    for line in game_map:
        # noinspection PyTypeChecker
        line_list = [cell.value / len(MapStatus) for cell in line]
        game_list.append(line_list)
    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    return torch.tensor([[game_list]], dtype=torch.float32)


def map2tensor(game_map):
    if game_map is None:
        return None
    elif isinstance(game_map, bool):
        return None
    return game2tensor(game_map, is_map=True)
