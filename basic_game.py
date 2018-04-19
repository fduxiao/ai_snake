import random
import enum


__all__ = ['Direction', 'Game', 'MapStatus']


class Direction(enum.Enum):
    NOPE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


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
    width: int
    height: int

    snake: list
    food: tuple

    direction: Direction
    result: None

    def __init__(self, width=80, height=24, x: int=None, y: int=None, direction=None):
        assert self.reset(width, height, x, y, direction)

    def reset(self, width, height, x: int=None, y: int=None, direction=None):
        self.width = width
        self.height = height
        self.snake = list()

        if x is None:
            x = random.randrange(width)
        else:
            assert x in range(width)

        if y is None:
            y = random.randrange(height)
        else:
            assert y in range(height)

        if direction is None:
            direction = Direction(random.randrange(4))
        else:
            direction = Direction(direction)
            assert direction.value in range(4)

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
        assert self.direction is not Direction.NOPE

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

    def next_step(self, direction=Direction.NOPE) -> MoveStatus:
        """

        :param direction: which direction to go
        :return: false if the snake hits something
                 true for success
        """
        if direction is not Direction.NOPE:
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

    def draw_map(self):
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
