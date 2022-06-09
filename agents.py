import abc
import argparse
import enum
import re

import numpy as np
import numpy.typing as npt


class Agent(abc.ABC):

    class Type(enum.Enum):
        MELEE = 9, 1
        RANGED = 25, 2

        def __init__(self, number_of_actions: int, range: int):
            enum.Enum.__init__(self)
            self.number_of_actions = number_of_actions
            self.range = range

    class Action(enum.Enum):
        (MELEE_MOVE_UP, MELEE_MOVE_LEFT, MELEE_DO_NOTHING, MELEE_MOVE_RIGHT,
         MELEE_MOVE_DOWN, MELEE_ATTACK_UP, MELEE_ATTACK_LEFT,
         MELEE_ATTACK_RIGHT, MELEE_ATTACK_DOWN) = range(9)
        (RANGED_MOVE_UP_UP, RANGED_MOVE_UP_LEFT, RANGED_MOVE_UP,
         RANGED_MOVE_UP_RIGHT, RANGED_MOVE_LEFT_LEFT, RANGED_MOVE_LEFT,
         RANGED_DO_NOTHING, RANGED_MOVE_RIGHT, RANGED_MOVE_RIGHT_RIGHT,
         RANGED_MOVE_DOWN_LEFT, RANGED_MOVE_DOWN, RANGED_MOVE_DOWN_RIGHT,
         RANGED_MOVE_DOWN_DOWN, RANGED_ATTACK_UP_UP, RANGED_ATTACK_UP_LEFT,
         RANGED_ATTACK_UP, RANGED_ATTACK_UP_RIGHT, RANGED_ATTACK_LEFT_LEFT,
         RANGED_ATTACK_LEFT, RANGED_ATTACK_RIGHT, RANGED_ATTACK_RIGHT_RIGHT,
         RANGED_ATTACK_DOWN_LEFT, RANGED_ATTACK_DOWN, RANGED_ATTACK_DOWN_RIGHT,
         RANGED_ATTACK_DOWN_DOWN) = range(25)

    def __init__(self, args: argparse.Namespace, name: str):
        self.args = args
        self.name = name
        match = re.search(r'^(red|blue)(melee?|ranged)_(\d+)$', self.name)
        self.team = match.group(1)
        agent_type = match.group(2).upper()
        # petting zoo environment doesn't return bluemelee_X but bluemele_X
        if agent_type.endswith('LE'): agent_type += 'E'
        self.type = Agent.Type[agent_type]
        self.number = int(match.group(3))

        self.observation = None
        self.reward = None
        self.done = None
        self.info = None

    def see(self, observation: npt.NDArray, reward: int, done: bool,
            info: dict):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

    @abc.abstractmethod
    def action(self) -> int:
        """
        Should be runned after function `Agent.see(...)` and based on 
        the observation return the index of one of the actions
        """
        raise NotImplementedError()


class RandomAgent(Agent):

    def action(self) -> int:
        if self.done:  # necessary
            return None
        return np.random.randint(self.type.number_of_actions)


class DoNothingAgent(Agent):

    def action(self) -> int:
        if self.done:  # necessary
            return None
        return Agent.Action.RANGED_DO_NOTHING.value if self.type == Agent.Type.RANGED else Agent.Action.MELEE_DO_NOTHING.value


class GreedyAgent(Agent):

    POSITION = np.array([6, 6])

    def action(self) -> int:
        """
        If there is any enemy in the `observation` then: if it is in range attacks it, otherwise moves towards it

        Problems
        --------
        - The channel 3 in the `observation` doens't seem to show the correct enemies
        - Doesn't "see" if there is something where he wants to go, if there is he doesn't move
        """
        if self.done:  # necessary
            print(f'agent {self.name} died, returning None as action')
            return None

        print(f'observation.shape = {self.observation.shape}')
        for i in range(self.observation.shape[-1]):
            print(f'Channel {i}:\n{self.observation[:, :, i]}')
        print(f'REWARD: {self.reward}')
        print(f'agent name: {self.name}')

        enemy_presence_index = 4 if self.args.env_minimap_mode else 3
        enemy_positions = np.array(
            np.where(self.observation[:, :, enemy_presence_index] == 1))
        print(f'enemy positions:\n{enemy_positions}')

        agent_action = self.type.name
        if enemy_positions.any():
            closest_enemy_index = closest_index(self.POSITION, enemy_positions)
            closest_enemy_position = enemy_positions[:, closest_enemy_index]
            print(f'closest enemy is in position: {closest_enemy_position}')

            closest_enemy_relative = closest_enemy_position - self.POSITION
            print(f'closest enemy relative position: {closest_enemy_relative}')

            agent_action += '_ATTACK' if self._can_attack(
                closest_enemy_position) else '_MOVE'
            x, y = closest_enemy_relative
            if self.type == Agent.Type.MELEE:
                if abs(x) > abs(y):
                    if x < 0:
                        agent_action += '_UP'
                    else:
                        agent_action += '_DOWN'
                else:
                    if y < 0:
                        agent_action += '_LEFT'
                    else:
                        agent_action += '_RIGHT'
            else:
                if x < 0:
                    agent_action += '_UP' * (1 if y != 0 else min(
                        -x, self.type.range))
                elif x > 0:
                    agent_action += '_DOWN' * (1 if y != 0 else min(
                        x, self.type.range))

                if y < 0:
                    agent_action += '_LEFT' * (1 if x != 0 else min(
                        -y, self.type.range))
                elif y > 0:
                    agent_action += '_RIGHT' * (1 if x != 0 else min(
                        y, self.type.range))
        else:  # TODO do something when there is no enemies on the observation view
            agent_action += '_MOVE'
            print('No enemies found, returning ', end='')
            if self.team == 'red':
                agent_action += '_RIGHT' * self.type.range
                print('right because I\'m on the red team')
            else:
                agent_action += '_LEFT' * self.type.range
                print('left because I\'m on the blue team')

        _action = Agent.Action[agent_action]
        print(f'Chosen action: {agent_action} ({_action.value})')
        return _action.value

    def _can_attack(self, enemy_position: npt.NDArray):
        distance = euclidean_distance(self.POSITION, enemy_position)[0]
        return distance <= self.type.range


def closest_index(point: npt.NDArray, points: npt.NDArray):
    """
    Returns the index of the closest point, in `points`, to the `point`

    Examples
    ----------
    >>> points = np.array([[6, 8, 10], [8, 8, 8]])
    array([[6, 8, 10],  # x's
           [8, 8, 8]])  # y's
    >>> point = np.array([6, 6])
    array([6, 6])
    >>> closest_index(point, points)
    0
    >>> point = np.array([10, 10])
    array([10, 10])
    >>> closest_index(point, points)
    2
    """
    if len(point.shape) == 1:
        point = point[:, np.newaxis]
    if len(points.shape) == 1:
        points = points[:, np.newaxis]
    return np.argmin(np.sum((points - point)**2, axis=0))


def euclidean_distance(point: npt.NDArray, points: npt.NDArray):
    """
    Returns a list with the euclidean distance from `point` to each point in `points`

    Examples
    ----------
    >>> points = np.array([[6, 8, 10], [8, 8, 8]])
    array([[6, 8, 10],  # x's
           [8, 8, 8]])  # y's
    >>> point = np.array([6, 6])
    array([6, 6])
    >>> euclidean_distance(point, points)
    array([2.        , 2.82842712, 4.47213595])
    >>> point = np.array([10, 10])
    array([10, 10])
    >>> euclidean_distance(point, points)
    array([4.47213595, 2.82842712, 2.        ])
    """
    if len(point.shape) == 1:
        point = point[:, np.newaxis]
    if len(points.shape) == 1:
        points = points[:, np.newaxis]
    return np.sum((points - point)**2, axis=0)**(1 / 2)


if __name__ == '__main__':
    agent = RandomAgent('blueranged_2')
    print(f'actions = {agent.action()}')

    print(f'AgentActions attributes = {Agent.Action.RANGED_MOVE_UP.value}')
