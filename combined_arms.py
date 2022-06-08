import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pettingzoo.magent.combined_arms_v6 as combined_arms
import time
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--map-size', type=int, default=16)
    parser.add_argument('--render',
                        default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--minimap-mode',
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--extra-features',
                        default=False,
                        action=argparse.BooleanOptionalAction)

    return parser.parse_args()


def stupid_policy(action):
    return action


def random_policy():
    number_of_actions = env.action_space(agent).n
    action = np.random.randint(number_of_actions)
    return action


MELEE_MOVE_UP, MELEE_MOVE_LEFT, MELEE_DO_NOTHING, MELEE_MOVE_RIGHT, MELEE_MOVE_DOWN, MELEE_ATTACK_UP, MELEE_ATTACK_LEFT, MELEE_ATTACK_RIGHT, MELEE_ATTACK_DOWN = range(
    9)

RANGED_MOVE_UP_UP, RANGED_MOVE_UP_LEFT, RANGED_MOVE_UP, RANGED_MOVE_UP_RIGHT, RANGED_MOVE_LEFT_LEFT, RANGED_MOVE_LEFT, RANGED_DO_NOTHING, RANGED_MOVE_RIGHT, RANGED_MOVE_RIGHT_RIGHT, RANGED_MOVE_LEFT_DOWN, RANGED_MOVE_DOWN, RANGED_MOVE_DOWN_RIGHT, RANGED_MOVE_DOWN_DOWN, RANGED_ATTACK_UP_UP, RANGED_ATTACK_UP_LEFT, RANGED_ATTACK_UP, RANGED_ATTACK_UP_RIGHT, RANGED_ATTACK_LEFT_LEFT, RANGED_ATTACK_LEFT, RANGED_ATTACK_RIGHT, RANGED_ATTACK_RIGHT_RIGHT, RANGED_ATTACK_LEFT_DOWN, RANGED_ATTACK_DOWN, RANGED_ATTACK_DOWN_RIGHT, RANGED_ATTACK_DOWN_DOWN = range(
    25)

if __name__ == '__main__':
    args = get_args()
    env = combined_arms.env(map_size=args.map_size,
                            minimap_mode=args.minimap_mode,
                            extra_features=args.extra_features)
    # print(f'action_spaces = {env.action_spaces}')

    list_of_results = []
    rewards_list = []
    list_of_episode_length = []

    agent_list = env.possible_agents
    agents_alive = agent_list

    #Initilize Pandas dataFrames
    df_aliveAtEnd = pd.DataFrame(columns=[
        "red_ranged_alive", "red_melee_alive", "red_tot_alive",
        "blue_ranged_alive", "blue_melee_alive", "blue_tot_alive"
    ])

    for episode in range(args.episodes):
        print(f'running episode {episode + 1}')
        env.reset()
        episode_length = 0

        for agent in env.agent_iter():
            observation, reward, done, info = env.last()

            print(f'observation.shape = {observation.shape}')
            for i in range(observation.shape[-1]):
                print(f'Channel {i}: {observation[:, :, i]}')
            print(f'REWARD: {reward}')
            print(f'agent: {agent}')

            if done:
                action = None  # necessary

            elif 'red' in agent:
                if 'mele' in agent:  #? RED in the render
                    action = stupid_policy(MELEE_DO_NOTHING)
                else:  # ranged #? BLUE in the render
                    action = stupid_policy(RANGED_MOVE_RIGHT_RIGHT)
            else:  # blue
                if 'mele' in agent:  #? GREEN in the render
                    action = stupid_policy(MELEE_MOVE_LEFT)
                else:  # ranged #? BLACK in the render
                    action = stupid_policy(RANGED_DO_NOTHING)
            print(f'action: {action}')

            if not done:
                agents_alive = env.agents
            env.step(action)

            if args.render:
                env.render()
                input('\nPress enter to continue...')

        #Making a dataframe of remaining agents
        redAgentsMeleeAlive = [
            i for i in agents_alive if i.startswith('redmelee')
        ]
        blueAgentsMeleeAlive = [
            i for i in agents_alive if i.startswith('bluemele')
        ]
        redAgentsRangedAlive = [
            i for i in agents_alive if i.startswith('redranged')
        ]
        blueAgentsRangedAlive = [
            i for i in agents_alive if i.startswith('blueranged')
        ]

        blueAgentsMeleeAlive_n = len(blueAgentsMeleeAlive)
        blueAgentsRangedAlive_n = len(blueAgentsRangedAlive)
        redAgentsMeleeAlive_n = len(redAgentsMeleeAlive)
        redAgentsRangedAlive_n = len(redAgentsRangedAlive)

        redTotAlive = redAgentsMeleeAlive_n + redAgentsRangedAlive_n
        blueTotAlive = blueAgentsMeleeAlive_n + blueAgentsRangedAlive_n

        series_alive = pd.Series([
            redAgentsRangedAlive_n, redAgentsMeleeAlive_n, redTotAlive,
            blueAgentsRangedAlive_n, blueAgentsMeleeAlive_n, blueTotAlive
        ],
                                 index=df_aliveAtEnd.columns)
        df_aliveAtEnd = df_aliveAtEnd.append(series_alive, ignore_index=True)

        list_of_episode_length.append(episode_length)

        #if args.render:
        #    env.render()
        #time.sleep(0.05)

    #Plotting the results from the dataframe
    x = [i for i in range(1, args.episodes + 1)]

    y_red_range = df_aliveAtEnd["red_ranged_alive"].tolist()
    y_red_melee = df_aliveAtEnd["red_melee_alive"].tolist()
    y_red_tot = df_aliveAtEnd["red_tot_alive"].tolist()

    y_blue_range = df_aliveAtEnd["blue_ranged_alive"].tolist()
    y_blue_melee = df_aliveAtEnd["blue_melee_alive"].tolist()
    y_blue_tot = df_aliveAtEnd["blue_tot_alive"].tolist()

    plt.plot(x,
             y_red_range,
             label="red_range",
             color="#ed0287",
             linestyle="dotted")
    plt.plot(x,
             y_red_melee,
             label="red_melee",
             color="#ff9900",
             linestyle="dashed")
    plt.plot(x, y_red_tot, label="red_tot", color="#ff0303")

    plt.plot(x,
             y_blue_range,
             label="blue_range",
             color="#3691b3",
             linestyle="dotted")
    plt.plot(x,
             y_blue_melee,
             label="blue_melee",
             color="#0b1563",
             linestyle="dashed")
    plt.plot(x, y_blue_tot, label="blue_tot", color="#0521f5")

    plt.xlabel('x - The episode')
    plt.ylabel('y - Number of agent left')
    plt.title("Different types of agents left in episodes")
    plt.legend()
    plt.show()

    #Seems like 12000/12 = 1000 is the default maximum iterations for each episode
    print(list_of_episode_length)