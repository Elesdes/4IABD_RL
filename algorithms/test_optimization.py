import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import itertools
import random
import csv
from do_not_touch.result_structures import PolicyAndActionValueFunction, PolicyAndValueFunction, ValueFunction
from do_not_touch.single_agent_env_wrapper import Env3
from envs.grid_world import GridWorldEnv, SingleAgentGridWorldEnv
from envs.line_world import LineWorldEnv, SingleAgentLineWorldEnv
from envs.tictactoe import TicTacToe


def sarsa_on_line_world(
        env: SingleAgentLineWorldEnv,
        gamma: float = 0.9999,
        alpha: float = 0.01,
        epsilon: float = 0.2,
        max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert epsilon > 0
    assert alpha > 0
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    steps_per_episode = np.zeros(max_episodes_count)
    episode_times = np.zeros(max_episodes_count)
    start_time = time.time()
    for ep_id in range(max_episodes_count):
        env.reset()
        steps = 0
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions()

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                best_a_idx = np.argmax(Q[s][aa])
                a = aa[best_a_idx]

            old_score = env.score()
            env.step(a)
            steps += 1
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions()

            # Watch out, you need to take a specific a' AND it needs to not be "game over"-like
            if aa_p:
                if np.random.random() < epsilon:
                    a_p = np.random.choice(aa_p)
                else:
                    best_a_p_idx = np.argmax(Q[s_p][aa_p])
                    a_p = aa[best_a_p_idx]

            if env.is_game_over():
                Q[s_p, :] = 0.0
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                Q[s, a] += alpha * (r + gamma * (Q[s_p][a_p]) - Q[s, a])

            pi[s, :] = 0.0
            pi[s, aa[np.argmax(Q[s][aa])]] = 1.0
        steps_per_episode[ep_id] = steps
        episode_times[ep_id] = time.time() - start_time

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q))), steps_per_episode, episode_times


def q_learning_on_line_world(
        env: SingleAgentLineWorldEnv,
        gamma: float = 0.9999,
        alpha: float = 0.01,
        epsilon: float = 0.2,
        max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert epsilon > 0
    assert alpha > 0
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    steps_per_episode = np.zeros(max_episodes_count)
    episode_times = np.zeros(max_episodes_count)
    start_time = time.time()
    for ep_id in range(max_episodes_count):
        env.reset()
        steps = 0
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions()

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                best_a_idx = np.argmax(Q[s][aa])
                a = aa[best_a_idx]

            old_score = env.score()
            env.step(a)
            steps += 1
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions()

            if env.is_game_over():
                Q[s_p, :] = 0.0
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                Q[s, a] += alpha * (r + gamma * np.max(Q[s_p][aa_p]) - Q[s, a])

            pi[s, :] = 0.0
            pi[s, aa[np.argmax(Q[s][aa])]] = 1.0

        steps_per_episode[ep_id] = steps
        episode_times[ep_id] = time.time() - start_time
    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q))), steps_per_episode, episode_times


def expected_sarsa_on_line_world(
        env: SingleAgentLineWorldEnv,
        gamma: float = 0.9999,
        alpha: float = 0.01,
        epsilon: float = 0.2,
        max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert epsilon > 0
    assert alpha > 0
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    steps_per_episode = np.zeros(max_episodes_count)
    episode_times = np.zeros(max_episodes_count)
    start_time = time.time()
    for ep_id in range(max_episodes_count):
        env.reset()
        steps = 0
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions()

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                best_a_idx = np.argmax(Q[s][aa])
                a = aa[best_a_idx]

            old_score = env.score()
            env.step(a)
            steps += 1
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions()

            if env.is_game_over():
                Q[s_p, :] = 0.0
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                expected_value = np.dot(Q[s_p][aa_p], pi[s_p][aa_p])
                Q[s, a] += alpha * (r + gamma * expected_value - Q[s, a])

            pi[s, :] = 0.0
            pi[s, aa[np.argmax(Q[s][aa])]] = 1.0
        steps_per_episode[ep_id] = steps
        episode_times[ep_id] = time.time() - start_time
    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q))), steps_per_episode, episode_times


def sarsa_on_grid_world(
        env: SingleAgentGridWorldEnv,
        gamma: float = 0.9999,
        alpha: float = 0.01,
        epsilon: float = 0.2,
        max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert epsilon > 0
    assert alpha > 0
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    steps_per_episode = np.zeros(max_episodes_count)
    episode_times = np.zeros(max_episodes_count)
    start_time = time.time()
    for ep_id in range(max_episodes_count):
        env.reset()
        steps = 0
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions()

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                best_a_idx = np.argmax(Q[s][aa])
                a = aa[best_a_idx]

            old_score = env.score()
            env.step(a)
            steps += 1
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions()

            # Watch out, you need to take a specific a' AND it needs to not be "game over"-like
            if aa_p:
                if np.random.random() < epsilon:
                    a_p = np.random.choice(aa_p)
                else:
                    best_a_p_idx = np.argmax(Q[s_p][aa_p])
                    a_p = aa_p[best_a_p_idx]

            if env.is_game_over():
                Q[s_p, :] = 0.0
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                Q[s, a] += alpha * (r + gamma * (Q[s_p][a_p]) - Q[s, a])

            pi[s, :] = 0.0
            pi[s, aa[np.argmax(Q[s][aa])]] = 1.0

        steps_per_episode[ep_id] = steps
        episode_times[ep_id] = time.time() - start_time
    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q))), steps_per_episode, episode_times


def q_learning_on_grid_world(
        env: SingleAgentGridWorldEnv,
        gamma: float = 0.9999,
        alpha: float = 0.01,
        epsilon: float = 0.2,
        max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert epsilon > 0
    assert alpha > 0
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    steps_per_episode = np.zeros(max_episodes_count)
    episode_times = np.zeros(max_episodes_count)
    start_time = time.time()
    for ep_id in range(max_episodes_count):
        env.reset()
        steps = 0
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions()

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                best_a_idx = np.argmax(Q[s][aa])
                a = aa[best_a_idx]

            old_score = env.score()
            env.step(a)
            steps += 1
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions()

            if env.is_game_over():
                Q[s_p, :] = 0.0
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                Q[s, a] += alpha * (r + gamma * np.max(Q[s_p][aa_p]) - Q[s, a])

            pi[s, :] = 0.0
            pi[s, aa[np.argmax(Q[s][aa])]] = 1.0
        steps_per_episode[ep_id] = steps
        episode_times[ep_id] = time.time() - start_time
    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q))), steps_per_episode, episode_times


def expected_sarsa_on_grid_world(
        env: SingleAgentGridWorldEnv,
        gamma: float = 0.9999,
        alpha: float = 0.01,
        epsilon: float = 0.2,
        max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert epsilon > 0
    assert alpha > 0
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    steps_per_episode = np.zeros(max_episodes_count)
    episode_times = np.zeros(max_episodes_count)
    start_time = time.time()
    for ep_id in range(max_episodes_count):
        env.reset()
        steps = 0
        while not env.is_game_over():
            s = env.state_id()
            aa = env.available_actions()

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                best_a_idx = np.argmax(Q[s][aa])
                a = aa[best_a_idx]

            old_score = env.score()
            env.step(a)
            steps += 1
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions()

            if env.is_game_over():
                Q[s_p, :] = 0.0
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                expected_value = np.dot(Q[s_p][aa_p], pi[s_p][aa_p])
                Q[s, a] += alpha * (r + gamma * expected_value - Q[s, a])

            pi[s, :] = 0.0
            pi[s, aa[np.argmax(Q[s][aa])]] = 1.0
        steps_per_episode[ep_id] = steps
        episode_times[ep_id] = time.time() - start_time
    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q))), steps_per_episode, episode_times


def sarsa_on_tic_tac_toe_solo(env: TicTacToe,
                              gamma: float = 0.9999,
                              alpha: float = 0.01,
                              epsilon: float = 0.2,
                              max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert epsilon > 0
    assert alpha > 0
    pi = {}
    Q = {}
    possibilities = list(itertools.product([-1, 0, 1], repeat=9))
    dict_act = {i: random.uniform(-1, 1) for i in range(9)}
    dict_act_zeros = {i: 0 for i in range(9)}
    for i in possibilities:
        Q[i] = dict_act.copy()
        pi[i] = dict_act_zeros.copy()

    steps_per_episode = np.zeros(max_episodes_count)
    episode_times = np.zeros(max_episodes_count)
    start_time = time.time()
    for ep_id in range(max_episodes_count):
        env.reset_random()
        steps = 0
        while not env.is_game_over():
            s = [int(x) for x in env.state_id().replace('[', '').replace(']', '').strip().replace('  ', ' ').split(' ')]
            aa = env.available_actions_ids()
            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                take_Q = Q[tuple(s)]
                best_a_idx = -1
                temp_value = -1
                for x in take_Q.keys():
                    if Q[tuple(s)][x] > temp_value and x in aa:
                        temp_value = Q[tuple(s)][x]
                        best_a_idx = x
                a = best_a_idx

            old_score = env.score()
            env.act_with_action_id(a)
            steps += 1
            new_score = env.score()
            r = new_score - old_score
            s_p = [int(x) for x in
                   env.state_id().replace('[', '').replace(']', '').strip().replace('  ', ' ').split(' ')]
            if aa_p := env.available_actions_ids():
                if np.random.random() < epsilon:
                    a_p = np.random.choice(aa_p)
                else:
                    take_Q = Q[tuple(s_p)]
                    best_a_p_idx = -1
                    temp_value = -1
                    for x in take_Q.keys():
                        if Q[tuple(s_p)][x] > temp_value and x in aa_p:
                            temp_value = Q[tuple(s_p)][x]
                            best_a_p_idx = x
                    a_p = best_a_p_idx

            if env.is_game_over():
                for k in Q[tuple(s_p)]:
                    Q[tuple(s_p)][k] = 0.0
                Q[tuple(s)][a] += alpha * (r - Q[tuple(s)][a])
            else:
                Q[tuple(s)][a] += alpha * (r + gamma * (Q[tuple(s_p)][a_p]) - Q[tuple(s)][a])

            for k in pi[tuple(s)]:
                pi[tuple(s)][k] = 0.0
            take_Q = Q[tuple(s)]
            best_idx = -1
            temp_value = -1
            for x in take_Q.keys():
                if Q[tuple(s)][x] > temp_value and x in aa:
                    temp_value = Q[tuple(s)][x]
                    best_idx = x
            pi[tuple(s)][best_idx] = 1.0

        steps_per_episode[ep_id] = steps
        episode_times[ep_id] = time.time() - start_time
    return PolicyAndActionValueFunction(pi={i: value for i, value in enumerate(pi.values())},
                                        q={i: value for i, value in enumerate(Q.values())}), steps_per_episode, episode_times


def q_learning_on_tic_tac_toe_solo(
    env: TicTacToe,
    gamma: float = 0.9999,
    alpha: float = 0.01,
    epsilon: float = 0.2,
    max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert epsilon > 0
    assert alpha > 0
    pi = {}
    Q = {}
    possibilities = list(itertools.product([-1, 0, 1], repeat=9))
    dict_act = {i: random.uniform(-1, 1) for i in range(9)}
    dict_act_zeros = {i: 0 for i in range(9)}
    for i in possibilities:
        Q[i] = dict_act.copy()
        pi[i] = dict_act_zeros.copy()

    steps_per_episode = np.zeros(max_episodes_count)
    episode_times = np.zeros(max_episodes_count)
    start_time = time.time()
    for ep_id in range(max_episodes_count):
        env.reset_random()
        steps = 0
        while not env.is_game_over():
            s = [int(x) for x in env.state_id().replace('[', '').replace(']', '').strip().replace('  ', ' ').split(' ')]
            aa = env.available_actions_ids()

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                take_Q = Q[tuple(s)]
                best_a_idx = -1
                temp_value = -1
                for x in take_Q.keys():
                    if Q[tuple(s)][x] > temp_value and x in aa:
                        temp_value = Q[tuple(s)][x]
                        best_a_idx = x
                a = best_a_idx

            old_score = env.score()
            env.act_with_action_id(a)
            steps += 1
            new_score = env.score()
            r = new_score - old_score

            s_p = [int(x) for x in
                   env.state_id().replace('[', '').replace(']', '').strip().replace('  ', ' ').split(' ')]
            aa_p = env.available_actions_ids()

            if env.is_game_over():
                for k in Q[tuple(s_p)]:
                    Q[tuple(s_p)][k] = 0.0

                Q[tuple(s)][a] += alpha * (r - Q[tuple(s)][a])
            else:
                max_value = -1
                for key, value in Q[tuple(s_p)].items():
                    if value > max_value and key in aa_p:
                        max_value = value
                Q[tuple(s)][a] += alpha * (r + gamma * max_value - Q[tuple(s)][a])

            for k in pi[tuple(s)]:
                pi[tuple(s)][k] = 0.0

            take_Q = Q[tuple(s)]
            best_idx = -1
            temp_value = -1
            for x in take_Q.keys():
                if Q[tuple(s)][x] > temp_value and x in aa:
                    temp_value = Q[tuple(s)][x]
                    best_idx = x
            pi[tuple(s)][best_idx] = 1.0
        steps_per_episode[ep_id] = steps
        episode_times[ep_id] = time.time() - start_time
    return PolicyAndActionValueFunction(pi={i: value for i, value in enumerate(pi.values())},
                                        q={i: value for i, value in enumerate(Q.values())}), steps_per_episode, episode_times


def expected_sarsa_on_tic_tac_toe_solo(
    env: TicTacToe,
    gamma: float = 0.9999,
    alpha: float = 0.01,
    epsilon: float = 0.2,
    max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # Needs to do the np.dot manually only for this case
    def matmul(A, B):
        # Vérifier que les dimensions des matrices sont compatibles
        assert len(A) == len(B)
        list1 = list(A.values())
        list2 = list(B.values())
        # Effectuer la multiplication matricielle
        result = np.dot(list1, list2)
        return result


    assert epsilon > 0
    assert alpha > 0
    pi = {}
    Q = {}
    possibilities = list(itertools.product([-1, 0, 1], repeat=9))
    dict_act = {i: random.uniform(-1, 1) for i in range(9)}
    dict_act_zeros = {i: 0 for i in range(9)}
    for i in possibilities:
        Q[i] = dict_act.copy()
        pi[i] = dict_act_zeros.copy()

    steps_per_episode = np.zeros(max_episodes_count)
    episode_times = np.zeros(max_episodes_count)
    start_time = time.time()
    for ep_id in range(max_episodes_count):
        env.reset_random()
        steps = 0
        while not env.is_game_over():
            s = [int(x) for x in env.state_id().replace('[', '').replace(']', '').strip().replace('  ', ' ').split(' ')]
            aa = env.available_actions_ids()

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                take_Q = Q[tuple(s)]
                best_a_idx = -1
                temp_value = -1
                for x in take_Q.keys():
                    if Q[tuple(s)][x] > temp_value and x in aa:
                        temp_value = Q[tuple(s)][x]
                        best_a_idx = x
                a = best_a_idx

            old_score = env.score()
            env.act_with_action_id(a)
            steps += 1
            new_score = env.score()
            r = new_score - old_score

            s_p = [int(x) for x in
                   env.state_id().replace('[', '').replace(']', '').strip().replace('  ', ' ').split(' ')]
            aa_p = env.available_actions_ids()

            if env.is_game_over():
                for k in Q[tuple(s_p)]:
                    Q[tuple(s_p)][k] = 0.0
                Q[tuple(s)][a] += alpha * (r - Q[tuple(s)][a])
            else:
                take_Q = {k: v for k, v in Q[tuple(s_p)].copy().items() if k in aa_p}
                take_pi = {k: v for k, v in pi[tuple(s_p)].copy().items() if k in aa_p}
                expected_value = matmul(take_Q, take_pi)
                Q[tuple(s)][a] += alpha * (r + gamma * expected_value - Q[tuple(s)][a])

            for k in pi[tuple(s)]:
                pi[tuple(s)][k] = 0.0
            take_Q = Q[tuple(s)]
            best_idx = -1
            temp_value = -1
            for x in take_Q.keys():
                if Q[tuple(s)][x] > temp_value and x in aa:
                    temp_value = Q[tuple(s)][x]
                    best_idx = x
            pi[tuple(s)][best_idx] = 1.0
        steps_per_episode[ep_id] = steps
        episode_times[ep_id] = time.time() - start_time
    return PolicyAndActionValueFunction(pi={i: value for i, value in enumerate(pi.values())},
                                        q={i: value for i, value in enumerate(Q.values())}), steps_per_episode, episode_times


def sarsa_on_secret_env3(
        gamma: float = 0.9999,
        alpha: float = 0.01,
        epsilon: float = 0.2,
        max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    assert epsilon > 0
    assert alpha > 0

    Q = {}
    pi = {}

    def choose_action(state, available_actions):
        if np.random.random() < epsilon:
            action = np.random.choice(available_actions)
        else:
            best_action = max(Q[state], key=Q[state].get)
            action = best_action
        return action

    for ep_id in range(max_episodes_count):
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            aa = np.copy(env.available_actions_ids())

            if s not in Q:
                Q[s] = {}
                pi[s] = {}

            for a in aa:
                if a not in Q[s]:
                    Q[s][a] = 0.0
                    pi[s][a] = 1.0 / len(aa)

            a = choose_action(s, aa)
            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = np.copy(env.available_actions_ids())

            if s_p not in Q:
                Q[s_p] = {}
                pi[s_p] = {}

            for a_p in aa_p:
                if a_p not in Q[s_p]:
                    Q[s_p][a_p] = 0.0
                    pi[s_p][a_p] = 1.0 / len(aa_p)

            a_p = choose_action(s_p, aa_p)

            if env.is_game_over():
                for for_ind in Q[s_p]:
                    Q[s_p][for_ind] = 0.0
                Q[s][a] += alpha * (r - Q[s][a])
            else:
                Q[s][a] += alpha * (r + gamma * Q[s_p][a_p] - Q[s][a])

            for for_ind in pi[s]:
                pi[s][for_ind] = 0.0

            max_value = 0
            max_a = 0
            for enum_a, a in enumerate(aa):
                if Q[s][a] > max_value:
                    max_value = Q[s][a]
                    max_a = enum_a

            # pi[s, aa[np.argmax(Q[s][aa])]] = 1.0
            pi[s][aa[max_a]] = 1.0

    return PolicyAndActionValueFunction(pi=dict(pi), q=dict(Q))


def q_learning_on_secret_env3(
        gamma: float = 0.9999,
        alpha: float = 0.01,
        epsilon: float = 0.2,
        max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    assert epsilon > 0
    assert alpha > 0

    Q = {}
    pi = {}

    def choose_action(state, available_actions):
        if np.random.random() < epsilon:
            action = np.random.choice(available_actions)
        else:
            best_action = max(Q[state], key=Q[state].get)
            action = best_action
        return action

    for ep_id in range(max_episodes_count):
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            aa = np.copy(env.available_actions_ids())

            if s not in Q:
                Q[s] = {}
                pi[s] = {}

            for a in aa:
                if a not in Q[s]:
                    Q[s][a] = 0.0
                    pi[s][a] = 1.0 / len(aa)

            a = choose_action(s, aa)
            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = np.copy(env.available_actions_ids())
            if s_p not in Q:
                Q[s_p] = {}
                pi[s_p] = {}

            for a_p in aa_p:
                if a_p not in Q[s_p]:
                    Q[s_p][a_p] = 0.0
                    pi[s_p][a_p] = 1.0 / len(aa_p)

            if env.is_game_over():
                for for_ind in Q[s_p]:
                    Q[s_p][for_ind] = 0.0
                Q[s][a] += alpha * (r - Q[s][a])
            else:
                Q[s][a] += alpha * (r + gamma * max(Q[s_p].values()) - Q[s][a])

            for for_ind in pi[s]:
                pi[s][for_ind] = 0.0

            max_value = 0
            max_a = 0
            for enum_a, a in enumerate(aa):
                if Q[s][a] > max_value:
                    max_value = Q[s][a]
                    max_a = enum_a
            # pi[s, aa[np.argmax(Q[s][aa])]] = 1.0
            pi[s][aa[max_a]] = 1.0
            # pi[s, :] = 0.0
            # pi[s, aa[np.argmax(Q[s][aa])]] = 1.0

    return PolicyAndActionValueFunction(pi=dict(pi), q=dict(Q))


def expected_sarsa_on_secret_env3(
        gamma: float = 0.9999,
        alpha: float = 0.01,
        epsilon: float = 0.2,
        max_episodes_count: int = 10000,
) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    assert epsilon > 0
    assert alpha > 0

    Q = {}
    pi = {}

    def choose_action(state, available_actions):
        if np.random.random() < epsilon:
            return np.random.choice(available_actions)
        else:
            return max(Q[state], key=Q[state].get)

    for ep_id in range(max_episodes_count):
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            aa = np.copy(env.available_actions_ids())

            if s not in Q:
                Q[s] = {}
                pi[s] = {}

            for a in aa:
                if a not in Q[s]:
                    Q[s][a] = 0.0
                    pi[s][a] = 1.0 / len(aa)

            a = choose_action(s, aa)
            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = np.copy(env.available_actions_ids())

            if s_p not in Q:
                Q[s_p] = {}
                pi[s_p] = {}

            for a_p in aa_p:
                if a_p not in Q[s_p]:
                    Q[s_p][a_p] = 0.0
                    pi[s_p][a_p] = 1.0 / len(aa_p)

            expected_value = 0
            for a_p in aa_p:
                prob_a_p = 1.0 - epsilon + epsilon / len(aa_p) if a_p == max(Q[s_p], key=Q[s_p].get) else epsilon / len(
                    aa_p)
                expected_value += prob_a_p * Q[s_p][a_p]

            if env.is_game_over():
                Q[s][a] += alpha * (r - Q[s][a])
            else:
                Q[s][a] += alpha * (r + gamma * expected_value - Q[s][a])

            for for_ind in pi[s]:
                pi[s][for_ind] = 0.0

            max_value = 0
            max_a = 0
            for enum_a, a in enumerate(aa):
                if Q[s][a] > max_value:
                    max_value = Q[s][a]
                    max_a = enum_a
            # pi[s, aa[np.argmax(Q[s][aa])]] = 1.0
            pi[s][aa[max_a]] = 1.0
    return PolicyAndActionValueFunction(pi=dict(pi), q=dict(Q))

def policy_evaluation_on_line_world(
    env: SingleAgentLineWorldEnv, theta: float = 0.0000001
) -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    V = np.zeros((env.state_space(),))
    start_time = time.time()
    step = 0
    while True:
        step += 1
        delta = 0.0
        for s in range(env.state_space()):
            old_v = V[s]
            total = 0.0
            for a in env.available_actions():
                total_inter = 0.0
                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total_inter += env.p(s, a, s_p, r) * (
                            env.possible_score()[r] + 0.99999 * V[s_p]
                        )
                total_inter = env.pi(s, a) * total_inter
                total += total_inter
            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))
        if delta < theta:
            return dict(enumerate(V.flatten(), 1)), step, time.time() - start_time


def policy_iteration_on_line_world(
    env: SingleAgentLineWorldEnv, theta: float = 0.0000001
) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    V = np.zeros((env.state_space(),))
    pi = np.random.randint(0, len(env.available_actions()), env.state_space())
    start_time = time.time()
    step = 0
    while True:
        # policy evaluation
        while True:
            delta = 0
            for s in range(env.state_space()):
                old_v = V[s]

                total = 0.0

                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total += env.p(s, pi[s], s_p, r) * (
                            env.possible_score()[r] + 0.99999999 * V[s_p]
                        )
                V[s] = total
                delta = max(delta, np.abs(old_v - V[s]))
            if delta < theta:
                break

        # policy improvement
        policy_stable = True
        for s in range(env.state_space()):
            old_action = pi[s]

            best_a = None
            best_action_score = None
            for a in env.available_actions():
                total = 0.0
                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total += env.p(s, a, s_p, r) * (
                            env.possible_score()[r] + 0.999999 * V[s_p]
                        )
                if best_a is None or total > best_action_score:
                    best_a = a
                    best_action_score = total

            pi[s] = best_a
            if best_a != old_action:
                policy_stable = False

        step += 1
        if policy_stable:
            return PolicyAndValueFunction(
                pi=dict(enumerate(pi)), v=dict(enumerate(V.flatten(), 1))
            ), step, time.time() - start_time


def value_iteration_on_line_world(
    env: SingleAgentLineWorldEnv, theta: float = 0.0000001
) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    V = np.zeros((env.state_space(),))
    pi = np.random.randint(0, len(env.available_actions()), env.state_space())
    start_time = time.time()
    step = 0
    while True:
        delta = 0
        step += 1
        for s in range(env.state_space()):
            old_v = V[s]

            best_action_score = None
            best_a = None
            for a in env.available_actions():
                total = 0.0
                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total += env.p(s, a, s_p, r) * (
                            env.possible_score()[r] + 0.99999999 * V[s_p]
                        )

                if best_action_score is None or total > best_action_score:
                    best_action_score = total
                    best_a = a

            V[s] = best_action_score
            pi[s] = best_a
            delta = max(delta, np.abs(old_v - V[s]))
        if delta < theta:
            break

    return PolicyAndValueFunction(
        pi=dict(enumerate(pi)), v=dict(enumerate(V.flatten(), 1))
    ), step, time.time() - start_time


def policy_evaluation_on_grid_world(
    env: SingleAgentGridWorldEnv, theta: float = 0.0000001
) -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    V = np.zeros((env.state_space(),))
    start_time = time.time()
    step = 0
    while True:
        delta = 0.0
        step += 1
        for s in range(env.state_space()):
            old_v = V[s]
            total = 0.0
            for a in env.available_actions():
                total_inter = 0.0
                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total_inter += env.p(s, a, s_p, r) * (
                            env.possible_score()[r] + 0.99999 * V[s_p]
                        )
                total_inter = env.pi(s, a) * total_inter
                total += total_inter
            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))
        if delta < theta:
            return dict(enumerate(V.flatten(), 1)), step, time.time() - start_time


def policy_iteration_on_grid_world(
    env: SingleAgentGridWorldEnv, theta: float = 0.0000001
) -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    V = np.zeros((env.state_space(),))
    pi = np.random.randint(0, len(env.available_actions()), env.state_space())
    start_time = time.time()
    step = 0
    while True:
        # policy evaluation
        while True:
            delta = 0
            for s in range(env.state_space()):
                old_v = V[s]

                total = 0.0

                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total += env.p(s, pi[s], s_p, r) * (
                            env.possible_score()[r] + 0.99999999 * V[s_p]
                        )
                V[s] = total
                delta = max(delta, np.abs(old_v - V[s]))
            if delta < theta:
                break

        # policy improvement
        policy_stable = True
        for s in range(env.state_space()):
            old_action = pi[s]

            best_a = None
            best_action_score = None
            for a in env.available_actions():
                total = 0.0
                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total += env.p(s, a, s_p, r) * (
                            env.possible_score()[r] + 0.999999 * V[s_p]
                        )
                if best_a is None or total > best_action_score:
                    best_a = a
                    best_action_score = total

            pi[s] = best_a
            if best_a != old_action:
                policy_stable = False

        step +=1
        if policy_stable:
            return PolicyAndValueFunction(
                pi=dict(enumerate(pi)), v=dict(enumerate(V.flatten(), 1))
            ), step, time.time() - start_time


def value_iteration_on_grid_world(
    env: SingleAgentGridWorldEnv, theta: float = 0.0000001
) -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    V = np.zeros((env.state_space(),))
    pi = np.random.randint(0, len(env.available_actions()), env.state_space())
    start_time = time.time()
    step = 0
    while True:
        step += 1
        delta = 0
        for s in range(env.state_space()):
            old_v = V[s]

            best_action_score = None
            best_a = None
            for a in env.available_actions():
                total = 0.0
                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total += env.p(s, a, s_p, r) * (
                            env.possible_score()[r] + 0.99999999 * V[s_p]
                        )

                if best_action_score is None or total > best_action_score:
                    best_action_score = total
                    best_a = a

            V[s] = best_action_score
            pi[s] = best_a
            delta = max(delta, np.abs(old_v - V[s]))
        if delta < theta:
            break

    return PolicyAndValueFunction(
        pi=dict(enumerate(pi)), v=dict(enumerate(V.flatten(), 1))
    ), step, time.time() - start_time

def find_five_same_value_in_a_row(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        values = [int(row[1]) for row in reader]

    for i in range(len(values) - 4):
        if all(values[i + j] == values[i] for j in range(5)):
            return int(i) + 1

    return -1


if __name__ == '__main__':
    line_world = LineWorldEnv(7)
    grid_world = GridWorldEnv(5, 5)
    tic_tac_toe_solo = TicTacToe()
    current_env = None
    """
    for env in ["line_world","grid_world"]:
        if env == "line_world":
            current_env = line_world
        else:
            current_env = grid_world
        for algo in ["sarsa", "q_learning", "expected_sarsa"]:
            for seed in [0,1,2,3,4,5,6,7,8,9]:
                np.random.seed(seed)
                for gamma in [0.9999, 0.75, 0.5]:
                    for alpha in [0.01, 0.1]:
                        for epsilon in [0.1, 0.2, 0.5]:
                            temp, steps_total, times_total = getattr(sys.modules[__name__], f"{algo}_on_{env}")(current_env, gamma=gamma, alpha=alpha, epsilon=epsilon)
                            steps_total = [int(x) for x in steps_total]
                            times_total = [round(x, 4) for x in times_total]
                            plt.plot(steps_total, c='red')
                            plt.savefig(f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\seed-{seed}\\img\\steps_gamma-{gamma}_alpha-{alpha}_epsilon-{epsilon}.png")
                            plt.clf()
                            plt.plot(times_total, steps_total, c='green')
                            plt.savefig(f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\seed-{seed}\\img\\timespersteps_gamma-{gamma}_alpha-{alpha}_epsilon-{epsilon}.png")
                            pd.DataFrame({"time": times_total, "step": steps_total}).to_csv(f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\seed-{seed}\\csv\\gamma-{gamma}_alpha-{alpha}_epsilon-{epsilon}.csv", index=False)
    """
    """
    for env in ["line_world","grid_world"]:
        if env == "line_world":
            current_env = line_world
        else:
            current_env = grid_world
        for algo in ["policy_evaluation", "policy_iteration", "value_iteration"]:
            for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                np.random.seed(seed)
                for theta in [0.0000001, 0.000001, 0.00000001]:
                    temp, steps_total, times_total = getattr(sys.modules[__name__], f"{algo}_on_{env}")(
                        current_env, theta=theta)
                    pd.DataFrame({"time": [round(times_total, 4)], "step": [steps_total]}).to_csv(
                        f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\seed-{seed}\\csv\\theta-{theta}.csv",
                        index=False)
            list_dir = [x for x in os.listdir(f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}")]
            result_for_each_seeds = [[], [], [], [], [], [], [], [], [], []]
            for index, directory in enumerate(list_dir):
                csv_dir = [x for x in os.listdir(f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\{directory}\\csv")]
                result_table = {}
                for file in csv_dir:
                    with open(f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\{directory}\\csv\\" + file, 'r') as file:
                        reader = csv.reader(file)
                        next(reader)
                        id_found = [int(row[1]) for row in reader]
                    result_table[file] = [id_found]
                result_for_each_seeds[index] = [y[0] for x in result_table for y in result_table[x]]

            average_result_for_each_seeds = []
            for col in range(3):
                sum_of_col = 0
                for row in range(10):
                    sum_of_col += result_for_each_seeds[row][col]
                average_result_for_each_seeds.append(sum_of_col / 10)

            pd.DataFrame(columns=average_result_for_each_seeds).to_csv(f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\results\\{env}\\{algo}\\results.csv",index=False)
            print(average_result_for_each_seeds)
    """
    for env in ["tic_tac_toe_solo"]:
        current_env = tic_tac_toe_solo
        for algo in ["sarsa", "q_learning", "expected_sarsa"]:
            for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                np.random.seed(seed)
                for gamma in [0.9999, 0.75, 0.5]:
                    for alpha in [0.01, 0.1]:
                        for epsilon in [0.1, 0.2, 0.5]:
                            temp, steps_total, times_total = getattr(sys.modules[__name__], f"{algo}_on_{env}")(
                                current_env, gamma=gamma, alpha=alpha, epsilon=epsilon)
                            steps_total = [int(x) for x in steps_total]
                            times_total = [round(x, 4) for x in times_total]
                            plt.plot(steps_total, c='red')
                            plt.savefig(
                                f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\seed-{seed}\\img\\steps_gamma-{gamma}_alpha-{alpha}_epsilon-{epsilon}.png")
                            plt.clf()
                            plt.plot(times_total, steps_total, c='green')
                            plt.savefig(
                                f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\seed-{seed}\\img\\timespersteps_gamma-{gamma}_alpha-{alpha}_epsilon-{epsilon}.png")
                            pd.DataFrame({"time": times_total, "step": steps_total}).to_csv(
                                f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\seed-{seed}\\csv\\gamma-{gamma}_alpha-{alpha}_epsilon-{epsilon}.csv",
                                index=False)
            list_dir = [x for x in os.listdir(
                f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}")]
            result_for_each_seeds = [[], [], [], [], [], [], [], [], [], []]
            for index, directory in enumerate(list_dir):
                csv_dir = [x for x in os.listdir(
                    f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\{directory}\\csv")]
                result_table = {}
                for file in csv_dir:
                    with open(
                            f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\{env}\\{algo}\\{directory}\\csv\\" + file,
                            'r') as file:
                        reader = csv.reader(file)
                        next(reader)
                        id_found = [int(row[1]) for row in reader]
                    result_table[file] = [id_found]
                result_for_each_seeds[index] = [y[0] for x in result_table for y in result_table[x]]

            average_result_for_each_seeds = []
            for col in range(18):
                sum_of_col = 0
                for row in range(10):
                    sum_of_col += result_for_each_seeds[row][col]
                average_result_for_each_seeds.append(sum_of_col / 10)

            pd.DataFrame(columns=average_result_for_each_seeds).to_csv(
                f"C:\\Users\\Erwan\\Desktop\\Code\\Deep_Reinforcement_Learning\\save_optimization\\results\\{env}\\{algo}\\results.csv",
                index=False)
            print(average_result_for_each_seeds)
    """
    print(sarsa_on_secret_env3())
    print(q_learning_on_secret_env3())
    print(expected_sarsa_on_secret_env3())
    """
    print("End")
