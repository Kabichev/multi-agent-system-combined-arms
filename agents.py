import abc
import argparse
import enum
import re

import numpy as np
import numpy.typing as npt


class Action(enum.Enum):

    def __str__(self):
        return f'{self.name} ({self.value})'


class MeleeAction(Action):
    (MOVE_UP, MOVE_LEFT, DO_NOTHING, MOVE_RIGHT, MOVE_DOWN, ATTACK_UP,
     ATTACK_LEFT, ATTACK_RIGHT, ATTACK_DOWN) = range(9)


class RangedAction(Action):
    (MOVE_UP_UP, MOVE_UP_LEFT, MOVE_UP, MOVE_UP_RIGHT, MOVE_LEFT_LEFT,
     MOVE_LEFT, DO_NOTHING, MOVE_RIGHT, MOVE_RIGHT_RIGHT, MOVE_DOWN_LEFT,
     MOVE_DOWN, MOVE_DOWN_RIGHT, MOVE_DOWN_DOWN, ATTACK_UP_UP, ATTACK_UP_LEFT,
     ATTACK_UP, ATTACK_UP_RIGHT, ATTACK_LEFT_LEFT, ATTACK_LEFT, ATTACK_RIGHT,
     ATTACK_RIGHT_RIGHT, ATTACK_DOWN_LEFT, ATTACK_DOWN, ATTACK_DOWN_RIGHT,
     ATTACK_DOWN_DOWN) = range(25)


class Type(enum.Enum):

    MELEE = 1, MeleeAction
    RANGED = 2, RangedAction

    def __init__(self, range: int, action: Action):
        enum.Enum.__init__(self)
        self.range = range
        self.action = action


class Agent(abc.ABC):

    RELATIVE_POSITION = np.array([6, 6])

    def __init__(self, args: argparse.Namespace, name: str, safe: bool = False):
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

        self.safe = safe
        self.in_danger = False

        # channels
        if self.type == Type.MELEE:
            self.ally_channels = [1, 3]
            self.enemy_channels = [5, 7]
        else:
            self.ally_channels = [1, 7]
            self.enemy_channels = [3, 5]
        if self.args.env_minimap_mode:
            self.enemy_channels += 1
            self.ally_channels[1] += 1
        self.agents_channels = self.ally_channels + self.enemy_channels

    def letter(self):
        if self.team == 'red':
            if self.type == Type.MELEE:
                return 'R'
            else:
                return 'B'
        else:
            if self.type == Type.MELEE:
                return 'g'
            else:
                return 'b'

    def see(self, observation: npt.NDArray, reward: int, done: bool,
            info: dict):
        if observation is None and reward is None and done is None and info is None:  # is dead
            self.hp = 0.0
            self.done = True
            return

        self.observation = observation
        current_hp = observation[self.RELATIVE_POSITION[0],
                              self.RELATIVE_POSITION[1], 2]

        if current_hp < 0.3 and self.safe:
            self.in_danger = True
        else:
            self.in_danger = False

        self.hp = current_hp
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

    def _can_attack(self, enemy_position: npt.NDArray):
        distance = euclidean_distance(self.RELATIVE_POSITION,
                                      enemy_position)[0]
        return distance <= self.type.range

    def _is_too_close(self, enemy_position: npt.NDArray):
        distance = euclidean_distance(self.RELATIVE_POSITION,
                                      enemy_position)[0]
        return distance <= 2

    def _can_select_safe_action(self):
        if self.team == 'red':
            direction = -1
            direction_str = '_LEFT'
        else:
            direction = 1
            direction_str = '_RIGHT'
        if self.type == Type.MELEE:
            for agents_channel in self.agents_channels:
                if self.observation[6, 6 + direction, agents_channel] == 1:
                    return False, None
            agent_action = 'MOVE' + direction_str
            return True, self.type.action[agent_action]
        else:
            free_square = 3
            for agents_channel in self.agents_channels:
                if self.observation[6, 6 + direction * 2, agents_channel] == 0:
                    free_square -= 1
            if free_square:
                agent_action = 'MOVE' + direction_str * 2
                return True, self.type.action[agent_action]
            for row in range(5, 8):
                free_square = 3
                for agents_channel in self.agents_channels:
                    if self.observation[row, 6 + direction, agents_channel] == 0:
                        free_square -= 1
                if free_square:
                    if row == 6:
                        agent_action = 'MOVE'
                    elif row < 6:
                        agent_action = 'MOVE_UP'
                    else:
                        agent_action = 'MOVE_DOWN'
                    agent_action += direction_str
                    self.last_action = self.type.action[agent_action]
                    return True, self.last_action
            return False, None


class RandomAgent(Agent):

    def action(self) -> Action:
        if self.done:  # necessary
            self.last_action = None
        else:
            self.last_action = np.random.choice(list(self.type.action))
        return self.last_action


class DoNothingAgent(Agent):

    def action(self) -> Action:
        if self.done:  # necessary
            self.last_action = None
        else:
            self.last_action = self.type.action['DO_NOTHING']
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
            self.last_action = None
            return None

        # print(f'agent name: {self.name}')
        # for i in range(self.observation.shape[-1]):
        #     if i >= 1 and i <= 8:
        #         print(f'Channel {i}:\n{self.observation[:, :, i]}')
        all_enemy_positions = np.empty((2, 0), dtype=int)
        for enemy_channel in self.enemy_channels:
            enemy_positions = np.array(
                np.where(self.observation[:, :, enemy_channel] == 1))
            if enemy_positions.any():
                all_enemy_positions = np.concatenate(
                    (all_enemy_positions, enemy_positions), axis=1)
        # print(f'all_enemy_positions = {all_enemy_positions}')

        if all_enemy_positions.any():
            closest_enemy_index = closest_index(self.RELATIVE_POSITION,
                                                all_enemy_positions)
            closest_enemy_position = all_enemy_positions[:,
                                                         closest_enemy_index]
            # print(f'closest enemy is in position: {closest_enemy_position}')

            closest_enemy_relative = closest_enemy_position - self.RELATIVE_POSITION
            # print(f'closest enemy relative position: {closest_enemy_relative}')

            if self.in_danger and self._is_too_close(closest_enemy_position):
                a, self.last_action = self._can_select_safe_action()
                if a:
                    return self.last_action

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
        return self.last_action


class ClingyGreedyAgent(Agent):

    def action(self) -> Action:
        """
        Agent's move actions are limited to 1 square, regardless of range, in order to stay close to its teammates.

        If there is any enemy in the `observation` then: if it is in range attacks it, otherwise moves towards it

        If no enemies are in the `observation` but an ally is: if the ally is close moves forward, otherwise moves
        towards it
        """
        if self.done:  # necessary
            self.last_action = None
            return None

        # Greedy Half
        all_enemy_positions = np.empty((2, 0), dtype=int)
        for enemy_channel in self.enemy_channels:
            enemy_positions = np.array(
                np.where(self.observation[:, :, enemy_channel] == 1))
            if enemy_positions.any():
                all_enemy_positions = np.concatenate(
                    (all_enemy_positions, enemy_positions), axis=1)
        # print(f'all_enemy_positions = {all_enemy_positions}')

        if all_enemy_positions.any():
            closest_enemy_index = closest_index(self.RELATIVE_POSITION,
                                                all_enemy_positions)
            closest_enemy_position = all_enemy_positions[:,
                                     closest_enemy_index]
            # print(f'closest enemy is in position: {closest_enemy_position}')

            closest_enemy_relative = closest_enemy_position - self.RELATIVE_POSITION
            # print(f'closest enemy relative position: {closest_enemy_relative}')

            if self.in_danger and self._is_too_close(closest_enemy_position):
                a, self.last_action = self._can_select_safe_action()
                if a:
                    return self.last_action

            if self._can_attack(closest_enemy_position):
                # if it can attack, it proceeds the same as the greedy agent
                agent_action = 'ATTACK'
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
            else:
                # if it can't attack, it moves only 1 square towards the enemy, diagonals allowed for ranged agents
                agent_action = 'MOVE'
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
                        agent_action += '_UP'
                    elif x > 0:
                        agent_action += '_DOWN'

                    if y < 0:
                        agent_action += '_LEFT'
                    elif y > 0:
                        agent_action += '_RIGHT'

        # Clingy Half
        else:
            # check for the closest allies the same it did for enemies
            all_ally_positions = np.empty((2, 0), dtype=int)
            for ally_channel in self.ally_channels:
                ally_positions = np.array(
                    np.where(self.observation[:, :, ally_channel] == 1))
                for n in range(ally_positions.shape[1]):
                    # need to remove the ally at position (6,6), because it's the agent itself
                    if ally_positions[0][n] == 6 and ally_positions[1][n] == 6:
                        ally_positions = np.delete(ally_positions, n, axis=1)
                        break
                if ally_positions.any():
                    all_ally_positions = np.concatenate(
                        (all_ally_positions, ally_positions), axis=1)

            if all_ally_positions.any():
                # locate the closest ally
                # print("I see an ally", self.name)
                closest_ally_index = closest_index(self.RELATIVE_POSITION,
                                                   all_ally_positions)
                closest_ally_position = all_ally_positions[:, closest_ally_index]

                closest_ally_relative = closest_ally_position - self.RELATIVE_POSITION
                agent_action = 'MOVE'
                if self._is_not_alone(closest_ally_position): # True if the closest ally is 1 square away
                    # move forward 1 square
                    if self.team == 'red':
                        agent_action += '_RIGHT'
                        # print('right because I\'m on the red team')
                    else:
                        agent_action += '_LEFT'
                        # print('left because I\'m on the blue team')
                else:
                    # closest ally is more than 1 square away, move towards it, diagonals allowed for ranged agents
                    # print("Alone", self.name)
                    x, y = closest_ally_relative
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
            else:
                # move forward
                agent_action = 'MOVE'
                if self.team == 'red':
                    agent_action += '_RIGHT' * self.type.range
                    # print('right because I\'m on the red team')
                else:
                    agent_action += '_LEFT' * self.type.range
                    # print('left because I\'m on the blue team')

        self.last_action = self.type.action[agent_action]
        return self.last_action

    def _is_not_alone(self, ally_position: npt.NDArray):
        """
        Returns True if the closest ally in only 1 square away from the agent (diagonals included), False otherwise.
        """
        distance = euclidean_distance(self.RELATIVE_POSITION,
                                      ally_position)[0]
        return distance < 2


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
    #print(f'actions = {agent.action()}')

    #print(f'AgentActions attributes = {Agent.Type.RangedAction.MOVE_UP.value}')

