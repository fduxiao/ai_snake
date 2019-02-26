import random
import enum


__all__ = ['Direction', 'Game', 'MapStatus', 'MoveStatus']


class Direction(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NOPE = 4


class MapStatus(enum.Enum):
    BLANK = 0
    HEAD = 1
    UP = 2
    DOWN = 3
    LEFT = 4
    RIGHT = 5
    FOOD = 6


class MoveStatus(enum.Enum):
    DEAD = 0
    EATEN = 1
    ALIVE = 2


class Game:
    width = 0
    height = 0

    snake = None
    food = None

    direction = None
    result = None

    n_step = 0

    def copy(self):
        game = Game()
        game.width = self.width
        game.height = self.height
        game.snake = self.snake.copy()
        game.food = self.food
        game.direction = self.direction
        game.result = self.result
        game.n_step = self.n_step
        return game

    def __init__(self, width=80, height=24, x: int = None, y: int = None, direction=None):
        if not self.reset(width, height, x, y, direction):
            raise ValueError("The game should have at least two empty block")

    def reset(self, width=None, height=None, x: int = None, y: int = None, direction=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        self.width = width
        self.height = height
        self.snake = list()
        self.result = None
        self.n_step = 0

        if x is None:
            x = random.randrange(width)
        else:
            if x not in range(width):
                raise ValueError("Wrong x value {} with width {}".format(x, width))

        if y is None:
            y = random.randrange(height)
        else:
            if y not in range(height):
                raise ValueError("Wrong y value {} with height {}".format(y, height))

        if direction is None:
            direction = Direction(random.randrange(4))
        else:
            direction = Direction(direction)
            if direction.value not in range(4):
                raise ValueError("Unknown direction {}".format(direction))

        self.direction = direction
        self.snake.append((x, y))
        return self.new_food()

    def new_food(self) -> bool:
        available = list()
        for i in range(self.width):
            for j in range(self.height):
                if (i, j) not in self.snake:
                    available.append((i, j))
        if len(available) == 0:
            return False  # You win the game
        pos = random.sample(available, 1)[0]
        self.food = pos
        return True  # means the game should continue

    def move(self):
        assert isinstance(self.direction, Direction)
        # each the snake must move
        # so the direction should not be nope
        assert self.direction is not None

        head = self.snake[0]
        if self.direction is Direction.UP:
            head = (head[0], head[1] - 1)
        elif self.direction is Direction.DOWN:
            head = (head[0], head[1] + 1)
        elif self.direction is Direction.LEFT:
            head = (head[0] - 1, head[1])
        elif self.direction is Direction.RIGHT:
            head = (head[0] + 1, head[1])
        # add the new head
        self.snake.insert(0, head)
        # then remove the tail
        return self.snake.pop(-1)

    def next_step(self, direction=None) -> MoveStatus:
        """

        :param direction: which direction to go
        :return: false if the snake hits something
                 true for success
        """
        self.n_step += 1
        if direction is not None:
            self.direction = direction

        tail = self.move()
        head = self.snake[0]
        # hit the walls
        if head[0] not in range(self.width) or head[1] not in range(self.height):
            return MoveStatus.DEAD
        if head in self.snake[1:]:  # hit itself
            return MoveStatus.DEAD
        if head == self.food:  # meet the food
            self.snake.append(tail)  # grow up
            return MoveStatus.EATEN
        return MoveStatus.ALIVE

    def draw_map(self) -> [[MapStatus]]:
        the_map = list()

        # blank
        for y in range(self.height):
            line = list()
            for x in range(self.width):
                line.append(MapStatus.BLANK)
            the_map.append(line)

        # food
        the_map[self.food[1]][self.food[0]] = MapStatus.FOOD

        # snake
        head = self.snake[0]
        the_map[head[1]][head[0]] = MapStatus.HEAD
        prev = head
        for s in self.snake[1:]:
            dx = s[0] - prev[0]
            dy = s[1] - prev[1]
            ss = None
            if dx == 1:
                ss = MapStatus.LEFT
            elif dx == -1:
                ss = MapStatus.RIGHT
            elif dy == 1:
                ss = MapStatus.UP
            elif dy == -1:
                ss = MapStatus.DOWN
            the_map[s[1]][s[0]] = ss
            prev = s

        return the_map

    def play(self, decider: callable):
        self.result = None
        while True:
            yield self.draw_map()

            direction = decider(self)
            if direction is None:
                return

            move_status = self.next_step(direction)
            if move_status is MoveStatus.DEAD:
                self.result = False
                return False
            elif move_status is MoveStatus.EATEN:
                if not self.new_food():  # no spare space
                    self.result = True
                    return True
            else:
                pass

    def train(self, decider: callable):
        """
        This should give the state after the decider is called

        :param decider: actual decider
        :return: yield previous state, post state
        """
        while True:
            game_before = self.copy()

            direction = decider(self)
            move_status = self.next_step(direction)

            game_after = self.copy()

            if move_status is MoveStatus.DEAD:
                yield game_before, None
                return False
            elif move_status is MoveStatus.EATEN:
                if not self.new_food():  # no spare space
                    yield game_before, True
                    return True
            else:
                yield game_before, game_after

    def head2food(self):
        head = self.snake[0]
        food = self.food
        delta_x = head[0] - food[0]
        delta_y = head[1] - food[1]
        return (delta_x ** 2 + delta_y ** 2) ** 0.5

    @property
    def len_per_step(self):
        return len(self.snake) / self.n_step
