from basic_game import Game, MapStatus, Direction
import curses
import argparse


char_map = {
    MapStatus.HEAD: '@',
    MapStatus.UP: '^',
    MapStatus.DOWN: 'v',
    MapStatus.LEFT: '<',
    MapStatus.RIGHT: '>',
    MapStatus.BLANK: ' ',
    MapStatus.FOOD: 'o',
}


def display_game(game_map: list, stdscr):
    stdscr.clear()

    stdscr.addstr('#')
    for _ in range(len(game_map[0])):
        stdscr.addstr('#')
    stdscr.addstr('#')
    stdscr.addstr('\n')

    for line in game_map:
        stdscr.addstr('#')
        for s in line:
            stdscr.addstr(char_map[s])
        stdscr.addstr('#\n')

    stdscr.addstr('#')
    for _ in range(len(game_map[0])):
        stdscr.addstr('#')
    stdscr.addstr('#')
    stdscr.addstr('\n')

    stdscr.refresh()


def interactive_decider(stdscr):
    rel_map = {
        curses.KEY_UP: Direction.UP,
        curses.KEY_DOWN: Direction.DOWN,
        curses.KEY_LEFT: Direction.LEFT,
        curses.KEY_RIGHT: Direction.RIGHT,
    }
    while True:
        d = stdscr.getch()
        if d not in rel_map:
            continue
        return rel_map.get(d, Direction.NOPE)


def main(stdscr):
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--width', type=int, default=20)
    parser.add_argument('-l', '--height', type=int, default=10)
    args = parser.parse_args()

    game = Game(width=args.width, height=args.height)

    for game_map in game.play(lambda _: interactive_decider(stdscr)):
        display_game(game_map, stdscr)
    if game.result is True:
        stdscr.addstr("Success!")
        display_game(game.draw_map(), stdscr)
    elif game.result is False:
        stdscr.addstr("Failure!")
    else:
        stdscr.addstr("Unknown error")
    stdscr.addstr("\npress any key to exit")
    stdscr.getkey()


if __name__ == '__main__':
    cur = __import__('curses')
    cur.start_color = lambda: None
    cur.wrapper(main)
