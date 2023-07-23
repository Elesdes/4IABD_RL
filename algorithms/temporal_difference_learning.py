# from algorithms.utils import return_policy_into_dict
import numpy as np
import random
import itertools
from do_not_touch.result_structures import PolicyAndActionValueFunction
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

    for _ in range(max_episodes_count):
        env.reset()
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
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            if aa_p := env.available_actions():
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

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q)))


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

    for _ in range(max_episodes_count):
        env.reset()
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

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q)))


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

    for _ in range(max_episodes_count):
        env.reset()
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

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q)))


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

    for _ in range(max_episodes_count):
        env.reset()
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
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            if aa_p := env.available_actions():
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

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q)))


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

    for _ in range(max_episodes_count):
        env.reset()
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

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q)))


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

    for _ in range(max_episodes_count):
        env.reset()
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

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q)))


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
    for _ in range(max_episodes_count):
        env.reset_random()
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

    return PolicyAndActionValueFunction(pi={i: value for i, value in enumerate(pi.values())},
                                        q={i: value for i, value in enumerate(Q.values())})


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

    for _ in range(max_episodes_count):
        env.reset_random()
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
    return PolicyAndActionValueFunction(pi={i: value for i, value in enumerate(pi.values())},
                                        q={i: value for i, value in enumerate(Q.values())})


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

    for _ in range(max_episodes_count):
        env.reset_random()
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

    return PolicyAndActionValueFunction(pi={i: value for i, value in enumerate(pi.values())},
                                        q={i: value for i, value in enumerate(Q.values())})


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

def test(
        env: TicTacToe,
        gamma: float = 0.9999,
        alpha: float = 0.1,
        epsilon: float = 0.2,
        max_episodes_count: int = 100000,
) -> PolicyAndActionValueFunction:
    assert epsilon > 0
    assert alpha > 0
    #pi = np.zeros((env.state_space(), env.action_space()))
    #Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))
    pi = {}
    Q = {}
    possibilities = list(itertools.product([-1, 0, 1], repeat=9))
    dict_act = {i: random.uniform(-1, 1) for i in range(9)}
    dict_act_zeros = {i: 0 for i in range(9)}
    for i in possibilities:
        Q[i] = dict_act.copy()
        pi[i] = dict_act_zeros.copy()
    #Q = np.array([list(Q[key].values()) for key in Q.keys()])
    #pi = np.array([list(pi[key].values()) for key in pi.keys()])
    for _ in range(max_episodes_count):
        env.reset_random()
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
                #best_a_idx = np.argmax(Q[s][aa])
                a = best_a_idx

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score
            s_p = [int(x) for x in env.state_id().replace('[', '').replace(']', '').strip().replace('  ', ' ').split(' ')]
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
                    #best_a_p_idx = np.argmax(Q[s_p][aa_p])
                    a_p = best_a_p_idx

            if env.is_game_over():
                for k in Q[tuple(s_p)]:
                    Q[tuple(s_p)][k] = 0.0
                #Q[tuple(s_p)][:] = 0.0
                Q[tuple(s)][a] += alpha * (r - Q[tuple(s)][a])
            else:
                Q[tuple(s)][a] += alpha * (r + gamma * (Q[tuple(s_p)][a_p]) - Q[tuple(s)][a])

            for k in pi[tuple(s)]:
                pi[tuple(s)][k] = 0.0
            #pi[tuple(s)][:] = 0.0
            take_Q = Q[tuple(s)]
            best_idx = -1
            temp_value = -1
            for x in take_Q.keys():
                if Q[tuple(s)][x] > temp_value and x in aa:
                    temp_value = Q[tuple(s)][x]
                    best_idx = x
            pi[tuple(s)][best_idx] = 1.0

    return PolicyAndActionValueFunction(pi={i: value for i, value in enumerate(pi.values())}, q={i: value for i, value in enumerate(Q.values())})


def demo():
    line_world = LineWorldEnv(7)
    grid_world = GridWorldEnv(5, 5)
    tictactoe = TicTacToe()
    """
    print(sarsa_on_line_world(line_world))
    print(q_learning_on_line_world(line_world))
    print(expected_sarsa_on_line_world(line_world))

    print(sarsa_on_grid_world(grid_world))
    print(q_learning_on_grid_world(grid_world))
    print(expected_sarsa_on_grid_world(grid_world))
    """

    #print(sarsa_on_tic_tac_toe_solo(tictactoe))
    #print(q_learning_on_tic_tac_toe_solo(tictactoe))
    print(expected_sarsa_on_tic_tac_toe_solo(tictactoe))

    # print(sarsa_on_secret_env3())
    # print(q_learning_on_secret_env3())
    # print(expected_sarsa_on_secret_env3())
    rule = expected_sarsa_on_tic_tac_toe_solo(tictactoe)
    possibilities = list(itertools.product([-1, 0, 1], repeat=9))
    #print(test(tictactoe))

    dict_tictactoe = {}
    for possibilitie, key in zip(possibilities, rule.pi.keys()):
        dict_tictactoe[possibilitie] = key
    tictactoe.reset_random()
    while not tictactoe.is_game_over():
        print(list(tictactoe.board))
        print(dict_tictactoe[tuple(list(tictactoe.board))])
        print(rule.pi[dict_tictactoe[tuple(list(tictactoe.board))]])
        choice = int(input("Choix: "))
        tictactoe.act_with_action_id(choice)
    print("End")


if __name__ == "__main__":
    demo()
