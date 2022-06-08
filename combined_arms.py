import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pettingzoo.magent.combined_arms_v6 as combined_arms
import time
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=n_of_episodes)
    parser.add_argument('--render',
                        default=True,
                        action=argparse.BooleanOptionalAction)

    return parser.parse_args()

def stupid_policy():

    action = 0
    return action

def random_policy():
    number_of_actions = env.action_space(agent).n
    action = np.random.randint(number_of_actions)
    return action



if __name__ == '__main__':
    n_of_episodes = 30
    args = get_args()
    env = combined_arms.env(map_size=16, minimap_mode=True)
    # print(f'action_spaces = {env.action_spaces}')

    list_of_results = []
    rewards_list = []
    list_of_episode_length = []

    agent_list = env.possible_agents
    agents_alive = agent_list

    #Initilize Pandas dataFrames
    df_aliveAtEnd = pd.DataFrame(columns=["red_ranged_alive", "red_melee_alive", "red_tot_alive",
                                            "blue_ranged_alive", "blue_melee_alive", "blue_tot_alive"])

    for episode in range(args.episodes):
        print(f'running episode {episode + 1}')
        env.reset()
        episode_length = 0

        for agent in env.agent_iter():
            observation, reward, done, info = env.last()

            if done:
                action = None  # necessary

            elif(agent.startswith("red")):
                action = random_policy()

            elif(agent.startswith("blue")):
                action = random_policy()

            if not done:
                agents_alive = env.agents
            env.step(action)

        #Making a dataframe of remaining agents
        redAgentsMeleeAlive = [i for i in agents_alive if i.startswith('redmelee')]
        blueAgentsMeleeAlive = [i for i in agents_alive if i.startswith('bluemele')]
        redAgentsRangedAlive = [i for i in agents_alive if i.startswith('redranged')]
        blueAgentsRangedAlive = [i for i in agents_alive if i.startswith('blueranged')]

        blueAgentsMeleeAlive_n = len(blueAgentsMeleeAlive)
        blueAgentsRangedAlive_n = len(blueAgentsRangedAlive)
        redAgentsMeleeAlive_n = len(redAgentsMeleeAlive)
        redAgentsRangedAlive_n = len(redAgentsRangedAlive)

        redTotAlive = redAgentsMeleeAlive_n + redAgentsRangedAlive_n
        blueTotAlive = blueAgentsMeleeAlive_n + blueAgentsRangedAlive_n

        series_alive = pd.Series([redAgentsRangedAlive_n, redAgentsMeleeAlive_n, redTotAlive,
                                  blueAgentsRangedAlive_n, blueAgentsMeleeAlive_n, blueTotAlive],
                                 index=df_aliveAtEnd.columns)
        df_aliveAtEnd = df_aliveAtEnd.append(series_alive, ignore_index=True)

        list_of_episode_length.append(episode_length)



            #if args.render:
            #    env.render()
                #time.sleep(0.05)

    #Plotting the results from the dataframe
    x = [i for i in range(1, n_of_episodes + 1)]

    y_red_range = df_aliveAtEnd["red_ranged_alive"].tolist()
    y_red_melee = df_aliveAtEnd["red_melee_alive"].tolist()
    y_red_tot = df_aliveAtEnd["red_tot_alive"].tolist()

    y_blue_range = df_aliveAtEnd["blue_ranged_alive"].tolist()
    y_blue_melee = df_aliveAtEnd["blue_melee_alive"].tolist()
    y_blue_tot = df_aliveAtEnd["blue_tot_alive"].tolist()

    plt.plot(x, y_red_range, label="red_range", color="#ed0287",
             linestyle="dotted"
             )
    plt.plot(x, y_red_melee, label="red_melee", color="#ff9900",
             linestyle="dashed")
    plt.plot(x, y_red_tot, label="red_tot", color="#ff0303")

    plt.plot(x, y_blue_range, label="blue_range", color="#3691b3",
             linestyle="dotted")
    plt.plot(x, y_blue_melee, label="blue_melee", color="#0b1563",
             linestyle="dashed")
    plt.plot(x, y_blue_tot, label="blue_tot", color="#0521f5")

    plt.xlabel('x - The episode')
    plt.ylabel('y - Number of agent left')
    plt.title("Different types of agents left in episodes")
    plt.legend()
    plt.show()

    #Seems like 12000/12 = 1000 is the default maximum iterations for each episode
    print(list_of_episode_length)