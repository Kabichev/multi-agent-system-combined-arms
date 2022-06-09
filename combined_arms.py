import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pettingzoo.magent.combined_arms_v6 as combined_arms
import pettingzoo.utils.env as pettingzoo_env

import agents


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int, default=30)
    parser.add_argument('-r',
                        '--render',
                        default=False,
                        action=argparse.BooleanOptionalAction)

    parser.add_argument('-ms', '--env-map-size', type=int, default=16)
    # defualt from environment
    parser.add_argument('-mm',
                        '--env-minimap-mode',
                        default=False,
                        action=argparse.BooleanOptionalAction)
    # defualt from environment
    parser.add_argument('-sr', '--env-step-reward', type=float, default=-.005)
    # defualt from environment
    parser.add_argument('-dp', '--env-dead-penalty', type=float, default=-.1)
    # defualt from environment
    parser.add_argument('-ap', '--env-attack-penalty', type=float, default=-.1)
    # defualt from environment
    parser.add_argument('-aor',
                        '--env-attack-opponent-reward',
                        type=float,
                        default=.2)
    # defualt from environment
    parser.add_argument('-mc', '--env-max-cycles', type=int, default=1000)
    parser.add_argument(
        '-ef',
        '--env-extra-features',
        default=False,
        action=argparse.BooleanOptionalAction)  # defualt from environment

    return parser.parse_args()


def reset_env(parallel_env: combined_arms.magent_parallel_env):
    observations = parallel_env.reset()

    all_agents, rewards, dones, infos = {}, {}, {}, {}
    for agent_name in parallel_env.agents:
        if 'blue' in agent_name:
            all_agents[agent_name] = agents.RandomAgent(args, agent_name)
        else:
            all_agents[agent_name] = agents.RandomAgent(args, agent_name)
        rewards[agent_name] = 0
        dones[agent_name] = False
        infos[agent_name] = {}

    return all_agents, observations, rewards, dones, infos


def render_env(args: argparse.Namespace,
               parallel_env: combined_arms.magent_parallel_env):
    if args.render:
        parallel_env.render()
        input('\nPress enter to continue...')  #! just for debug


def update_episodes_info(df: pd.DataFrame, agents_alive: list):
    redAgentsMeleeAlive = []
    blueAgentsMeleeAlive = []
    redAgentsRangedAlive = []
    blueAgentsRangedAlive = []
    for agent_name in agents_alive:
        if agent_name.startswith('redmelee'):
            redAgentsMeleeAlive += [agent_name]
        elif agent_name.startswith('bluemele'):
            blueAgentsMeleeAlive += [agent_name]
        elif agent_name.startswith('redranged'):
            redAgentsRangedAlive += [agent_name]
        elif agent_name.startswith('blueranged'):
            blueAgentsRangedAlive += [agent_name]

    blueAgentsMeleeAlive_n = len(blueAgentsMeleeAlive)
    blueAgentsRangedAlive_n = len(blueAgentsRangedAlive)
    redAgentsMeleeAlive_n = len(redAgentsMeleeAlive)
    redAgentsRangedAlive_n = len(redAgentsRangedAlive)

    redTotAlive = redAgentsMeleeAlive_n + redAgentsRangedAlive_n
    blueTotAlive = blueAgentsMeleeAlive_n + blueAgentsRangedAlive_n

    series_alive = pd.DataFrame([[
        redAgentsRangedAlive_n, redAgentsMeleeAlive_n, redTotAlive,
        blueAgentsRangedAlive_n, blueAgentsMeleeAlive_n, blueTotAlive
    ]],
                                columns=df.columns)
    return pd.concat([df, series_alive], ignore_index=True)


def plot_episodes_info(df: pd.DataFrame):
    x = [i for i in range(1, args.episodes + 1)]

    y_red_range = df["red_ranged_alive"].tolist()
    y_red_melee = df["red_melee_alive"].tolist()
    y_red_tot = df["red_tot_alive"].tolist()

    y_blue_range = df["blue_ranged_alive"].tolist()
    y_blue_melee = df["blue_melee_alive"].tolist()
    y_blue_tot = df["blue_tot_alive"].tolist()

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


if __name__ == '__main__':
    args = get_args()

    parallel_env = combined_arms.parallel_env(
        map_size=args.env_map_size,
        minimap_mode=args.env_minimap_mode,
        step_reward=args.env_step_reward,
        dead_penalty=args.env_dead_penalty,
        attack_penalty=args.env_attack_penalty,
        attack_opponent_reward=args.env_attack_opponent_reward,
        max_cycles=args.env_max_cycles,
        extra_features=args.env_extra_features)

    agent_list = parallel_env.possible_agents
    agents_alive = agent_list

    #Initilize Pandas dataFrames
    df_aliveAtEnd = pd.DataFrame(columns=[
        "red_ranged_alive", "red_melee_alive", "red_tot_alive",
        "blue_ranged_alive", "blue_melee_alive", "blue_tot_alive"
    ])

    for episode in range(args.episodes):
        print(f'running episode {episode + 1}')
        steps = 0
        all_agents, observations, rewards, dones, infos = reset_env(
            parallel_env)

        render_env(args, parallel_env)

        while steps < args.env_max_cycles:
            print(f'running step {steps + 1}')
            actions = {}
            for agent_name in parallel_env.agents:
                observation, reward, done, info = observations[
                    agent_name], rewards[agent_name], dones[agent_name], infos[
                        agent_name]
                all_agents[agent_name].see(observation, reward, done, info)
                actions[agent_name] = all_agents[agent_name].action()
            print(f'all actions: {actions}')
            observations, rewards, dones, infos = parallel_env.step(actions)
            steps += 1

            if np.array(list(dones.values())).all():
                break

            agents_alive = parallel_env.agents

            render_env(args, parallel_env)

        df_aliveAtEnd = update_episodes_info(df_aliveAtEnd, agents_alive)

    #Plotting the results from the dataframe
    plot_episodes_info(df_aliveAtEnd)