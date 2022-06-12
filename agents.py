import abc
import argparse
import enum
import re

import numpy as np
import numpy.typing as npt


class Action(enum.Enum):
    pass


class MeleeAction(Action):
    (MOVE_UP, MOVE_LEFT, DO_NOTHING, MOVE_RIGHT, MOVE_DOWN, ATTACK_UP,
     ATTACK_LEFT, ATTACK_RIGHT, ATTACK_DOWN) = range(9)


class RangedAction(Action):
    (MOVE_UP_UP, MOVE_UP_LEFT, MOVE_UP, MOVE_UP_RIGHT, MOVE_LEFT_LEFT,
     MOVE_LEFT, DO_NOTHING, MOVE_RIGHT, MOVE_RIGHT_RIGHT, MOVE_DOWN_LEFT,
     MOVE_DOWN, MOVE_DOWN_RIGHT, MOVE_DOWN_DOWN, ATTACK_UP_UP, ATTACK_UP_LEFT,
     ATTACK_UP, ATTACK_UP_RIGHT, ATTACK_LEFT_LEFT, ATTACK_LEFT, ATTACK_RIGHT,
     ATTACK_RIGHT_RIGHT, ATTACK_DOWN_LEFT, ATTACK_DOWN, ATTACK_DOWN_RIGHT,
     RANGED_ATTACK_DOWN_DOWN) = range(25)


class Type(enum.Enum):

    MELEE = 1, MeleeAction
    RANGED = 2, RangedAction

    def __init__(self, range: int, action: Action):
        enum.Enum.__init__(self)
        self.range = range
        self.action = action


class Agent(abc.ABC):

    RELATIVE_POSITION = np.array([6, 6])

    def __init__(self, args: argparse.Namespace, name: str):
        self.args = args
        self.name = name
        match = re.search(r'^(red|blue)(melee?|ranged)_(\d+)$', self.name)
        self.team = match.group(1)
        agent_type = match.group(2).upper()
        # petting zoo environment doesn't return bluemelee_X but bluemele_X
        if agent_type.endswith('LE'): agent_type += 'E'
        self.type = Type[agent_type]
        self.number = int(match.group(3))

        self.observation: npt.NDArray = None
        self.hp = 1.
        self.last_reward = .0
        self.done = False
        self.info: dict = None
        self.last_action: Action = None

    def see(self, observation: npt.NDArray, reward: int, done: bool,
            info: dict):
        self.observation = observation
        self.hp = observation[self.RELATIVE_POSITION[0],
                              self.RELATIVE_POSITION[1], 2]
        self.last_reward = reward
        self.done = done
        self.info = info

    @abc.abstractmethod
    def action(self) -> Action:
        """
        Should be runned after function `Agent.see(...)` and based on 
        the observation return the index of one of the actions
        """
        raise NotImplementedError()


class RandomAgent(Agent):

    def action(self) -> Action:
        if self.done:  # necessary
            self.last_action = None
        else:
            self.last_action = np.random.choice(list(self.type.action))
        print(f'Chosen action: {self.last_action.name} ({self.last_action.value})')
        return self.last_action


class DoNothingAgent(Agent):

    def action(self) -> Action:
        if self.done:  # necessary
            self.last_action = None
        else:
            self.last_action = self.type.action['DO_NOTHING']
        print(f'Chosen action: {self.last_action.name} ({self.last_action.value})')
        return self.last_action


class GreedyAgent(Agent):

    def action(self) -> Action:
        """
        If there is any enemy in the `observation` then: if it is in range attacks it, otherwise moves towards it

        Problems
        --------
        - The channel 3 in the `observation` doens't seem to show the correct enemies
        - Doesn't "see" if there is something where he wants to go, if there is he doesn't move
        """
        if self.done:  # necessary
            print(f'agent {self.name} died, returning None as action')
            self.last_action = None
            return None

        # print(f'observation.shape = {self.observation.shape}')
        # for i in range(self.observation.shape[-1]):
        #     print(f'Channel {i}:\n{self.observation[:, :, i]}')
        # print(f'REWARD: {self.reward}')
        # print(f'agent name: {self.name}')

        enemy_presence_index = 4 if self.args.env_minimap_mode else 3
        enemy_positions = np.array(
            np.where(self.observation[:, :, enemy_presence_index] == 1))
        # print(f'enemy positions:\n{enemy_positions}')

        if enemy_positions.any():
            closest_enemy_index = closest_index(self.RELATIVE_POSITION,
                                                enemy_positions)
            closest_enemy_position = enemy_positions[:, closest_enemy_index]
            # print(f'closest enemy is in position: {closest_enemy_position}')

            closest_enemy_relative = closest_enemy_position - self.RELATIVE_POSITION
            # print(f'closest enemy relative position: {closest_enemy_relative}')

            agent_action = 'ATTACK' if self._can_attack(
                closest_enemy_position) else 'MOVE'
            x, y = closest_enemy_relative
            if self.type == Type.MELEE:
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
            agent_action = 'MOVE'
            # print('No enemies found, returning ', end='')
            if self.team == 'red':
                agent_action += '_RIGHT' * self.type.range
                # print('right because I\'m on the red team')
            else:
                agent_action += '_LEFT' * self.type.range
                # print('left because I\'m on the blue team')

        self.last_action = self.type.action[agent_action]
        print(f'Chosen action: {self.last_action.name} ({self.last_action.value})')
        return self.last_action

    def _can_attack(self, enemy_position: npt.NDArray):
        distance = euclidean_distance(self.RELATIVE_POSITION,
                                      enemy_position)[0]
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

    print(f'AgentActions attributes = {Agent.Type.RangedAction.MOVE_UP.value}')
