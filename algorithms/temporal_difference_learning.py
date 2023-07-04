from do_not_touch.result_structures import PolicyAndActionValueFunction
from do_not_touch.single_agent_env_wrapper import Env3, Env2
from games.line_world import SingleAgentLineWorldEnv, LineWorldEnv
from games.grid_world import SingleAgentGridWorldEnv, GridWorldEnv
from games.tictactoe import TicTacToe
# from algorithms.utils import return_policy_into_dict
import numpy as np


def sarsa_on_line_world(env: SingleAgentLineWorldEnv,
                        gamma: float = 0.9999,
                        alpha: float = 0.01,
                        epsilon: float = 0.2,
                        max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert (epsilon > 0)
    assert (alpha > 0)
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    for ep_id in range(max_episodes_count):
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

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q)))


def q_learning_on_line_world(env: SingleAgentLineWorldEnv,
                             gamma: float = 0.9999,
                             alpha: float = 0.01,
                             epsilon: float = 0.2,
                             max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert (epsilon > 0)
    assert (alpha > 0)
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    for ep_id in range(max_episodes_count):
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


def expected_sarsa_on_line_world(env: SingleAgentLineWorldEnv,
                                 gamma: float = 0.9999,
                                 alpha: float = 0.01,
                                 epsilon: float = 0.2,
                                 max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert (epsilon > 0)
    assert (alpha > 0)
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    for ep_id in range(max_episodes_count):
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


def sarsa_on_grid_world(env: SingleAgentGridWorldEnv,
                        gamma: float = 0.9999,
                        alpha: float = 0.01,
                        epsilon: float = 0.2,
                        max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert (epsilon > 0)
    assert (alpha > 0)
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    for ep_id in range(max_episodes_count):
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

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q)))


def q_learning_on_grid_world(env: SingleAgentGridWorldEnv,
                             gamma: float = 0.9999,
                             alpha: float = 0.01,
                             epsilon: float = 0.2,
                             max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert (epsilon > 0)
    assert (alpha > 0)
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    for ep_id in range(max_episodes_count):
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


def expected_sarsa_on_grid_world(env: SingleAgentGridWorldEnv,
                                 gamma: float = 0.9999,
                                 alpha: float = 0.01,
                                 epsilon: float = 0.2,
                                 max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert (epsilon > 0)
    assert (alpha > 0)
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    for ep_id in range(max_episodes_count):
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
    assert (epsilon > 0)
    assert (alpha > 0)
    pi = np.zeros((9, 2))
    Q = np.random.uniform(-1.0, 1.0, (9, 2))

    for ep_id in range(max_episodes_count):
        env.reset()
        while not env.is_game_over():
            s = env.available_actions()
            aa = env.available_actions()
            """
            temp = env.available_actions()
            s = np.zeros(9, dtype=int)
            aa = np.zeros(9, dtype=int)
            for i_e, elem in enumerate(temp):
                s[i_e] = elem[0] * 3 + elem[1]
                aa[i_e] = elem[0] * 3 + elem[1]
            """
            if np.random.random() < epsilon:
                print(aa)
                a = np.random.choice(aa)
                print(a)
            else:
                temp_aa = np.zeros(len(aa), dtype=int)
                temp_s = np.zeros(len(s), dtype=int)
                for i_e, elem in enumerate(aa):
                    temp_aa[i_e] = elem[0] * 3 + elem[1]
                for i_e, elem in enumerate(s):
                    temp_s[i_e] = elem[0] * 3 + elem[1]
                best_a_idx = np.argmax(Q[temp_s][temp_aa])
                if best_a_idx%2==0:
                    best_a_idx = [int(best_a_idx/2), 0]
                else:
                    best_a_idx = [int((best_a_idx-1) / 2), 1]
                a = aa[best_a_idx]

            if env.is_game_over():
                old_score = -env.player
                env.play(a)
                new_score = -env.player
            else:
                old_score = 0
                env.play(a)
                if env.is_game_over():
                    new_score = env.player
                else:
                    new_score = 0
            r = new_score - old_score

            s_p = env.available_actions()
            aa_p = env.available_actions()

            # Watch out, you need to take a specific a' AND it needs to not be "game over"-like
            if len(aa_p) > 0:
                if np.random.random() < epsilon:
                    a_p = np.random.choice(aa_p)
                else:
                    temp_aa = np.zeros(len(aa_p), dtype=int)
                    temp_s = np.zeros(len(s_p), dtype=int)
                    for i_e, elem in enumerate(aa_p):
                        temp_aa[i_e] = elem[0] * 3 + elem[1]
                    for i_e, elem in enumerate(s_p):
                        temp_s[i_e] = elem[0] * 3 + elem[1]
                    print(Q)
                    print(temp_s)
                    print(temp_aa)
                    best_a_p_idx = np.argmax(Q[temp_s][temp_aa])
                    if best_a_p_idx % 2 == 0:
                        best_a_p_idx = [int(best_a_p_idx / 2), 0]
                    else:
                        best_a_p_idx = [int((best_a_p_idx - 1) / 2), 1]
                    a_p = aa_p[best_a_p_idx]

            if env.is_game_over():
                Q[s_p, :] = 0.0
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                Q[s, a] += alpha * (r + gamma * (Q[s_p][a_p]) - Q[s, a])

            pi[s, :] = 0.0
            pi[s, aa[np.argmax(Q[s][aa])]] = 1.0

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q)))


def q_learning_on_tic_tac_toe_solo(env: TicTacToe,
                                   gamma: float = 0.9999,
                                   alpha: float = 0.01,
                                   epsilon: float = 0.2,
                                   max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert (epsilon > 0)
    assert (alpha > 0)
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    for ep_id in range(max_episodes_count):
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


def expected_sarsa_on_tic_tac_toe_solo(env: TicTacToe,
                                       gamma: float = 0.9999,
                                       alpha: float = 0.01,
                                       epsilon: float = 0.2,
                                       max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    assert (epsilon > 0)
    assert (alpha > 0)
    pi = np.zeros((env.state_space(), env.action_space()))
    Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))

    for ep_id in range(max_episodes_count):
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


def sarsa_on_secret_env3(gamma: float = 0.9999,
                         alpha: float = 0.01,
                         epsilon: float = 0.2,
                         max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    assert (epsilon > 0)
    assert (alpha > 0)

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
        s = env.state_id()
        aa = env.available_actions_ids()

        if s not in Q:
            Q[s] = {}
            pi[s] = {}

        for a in aa:
            if a not in Q[s]:
                Q[s][a] = 0.0
                pi[s][a] = 1.0 / len(aa)

        while not env.is_game_over():
            a = choose_action(s, aa)
            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions_ids()

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
            max_ind = 0
            for ind_enum, ind_aa in enumerate(aa):
                if Q[s][ind_aa] > max_value:
                    max_value = Q[s][ind_aa]
                    max_ind = ind_enum
            pi[s][aa[max_ind]] = 1.0

            # pi[s, aa[np.argmax(Q[s][aa])]] = 1.0

    return PolicyAndActionValueFunction(pi=dict(pi), q=dict(Q))


def q_learning_on_secret_env3(gamma: float = 0.9999,
                              alpha: float = 0.01,
                              epsilon: float = 0.2,
                              max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    assert (epsilon > 0)
    assert (alpha > 0)

    # pi = np.zeros((env.state_space(), env.action_space()))
    # Q = np.random.uniform(-1.0, 1.0, (env.state_space(), env.action_space()))
    pi = np.empty(0)
    Q = np.empty(0)

    for ep_id in range(max_episodes_count):
        env.reset()
        while not env.is_game_over():
            pi_dict = dict(((j, i), pi[i][j]) for i in range(len(pi)) for j in range(len(pi[0])) if i < j)
            Q_dict = dict(((j, i), Q[i][j]) for i in range(len(Q)) for j in range(len(Q[0])) if i < j)

            s = env.state_id()
            aa = env.available_actions_ids()

            if s not in Q_dict:
                Q_dict[s] = {}
                for single_action in aa:
                    Q_dict[s][single_action] = 0.0

            if s not in pi_dict:
                pi_dict[s] = {}
                for single_action in aa:
                    pi_dict[s][single_action] = 0.0

            pi_dict = {0: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0},
                       1: {0: 0.1, 10: 0.4}}
            pi = np.array([list(pi_dict[key].values()) for key in pi_dict])
            Q = np.array([list(Q_dict[key].values()) for key in Q_dict])

            if np.random.random() < epsilon:
                a = np.random.choice(aa)
            else:
                best_a_idx = np.argmax(Q[s][aa])
                a = aa[best_a_idx]

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions_ids()

            if s_p not in Q_dict:
                Q_dict[s_p] = {}
                for single_action in aa_p:
                    Q_dict[s_p][single_action] = 0.0

            if s_p not in pi_dict:
                pi_dict[s_p] = {}
                for single_action in aa_p:
                    pi_dict[s_p][single_action] = 0.0

            pi = np.array([list(pi_dict[key].values()) for key in pi_dict])
            Q = np.array([list(Q_dict[key].values()) for key in Q_dict])
            print(Q)

            if env.is_game_over():
                Q[s_p, :] = 0.0
                Q[s][a] += alpha * (r - Q[s][a])
            else:
                try:
                    Q[s][a] += alpha * (r + gamma * np.max(Q[s_p][aa_p]) - Q[s][a])
                except IndexError:
                    print(s_p, aa_p)

            pi[s, :] = 0.0
            pi[s, aa[np.argmax(Q[s][aa])]] = 1.0

    return PolicyAndActionValueFunction(pi=dict(enumerate(pi)), q=dict(enumerate(Q)))


def expected_sarsa_on_secret_env3(gamma: float = 0.9999,
                                  alpha: float = 0.01,
                                  epsilon: float = 0.2,
                                  max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    assert (epsilon > 0)
    assert (alpha > 0)

    Q = {}
    pi = {}

    def choose_action(state, available_actions):
        if np.random.random() < epsilon:
            return np.random.choice(available_actions)
        else:
            return max(Q[state], key=Q[state].get)

    for ep_id in range(max_episodes_count):
        env.reset()
        s = env.state_id()
        aa = env.available_actions_ids()

        if s not in Q:
            Q[s] = {}
            pi[s] = {}

        for a in aa:
            if a not in Q[s]:
                Q[s][a] = 0.0
                pi[s][a] = 1.0 / len(aa)

        a = choose_action(s, aa)

        while not env.is_game_over():
            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions_ids()

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
            print(aa)
            for a in aa:
                if Q[s][a] > max_value:
                    max_a = a
            # pi[s, aa[np.argmax(Q[s][aa])]] = 1.0
            pi[s][aa[max_a]] = 1.0
    return PolicyAndActionValueFunction(pi=dict(pi), q=dict(Q))


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

    print(sarsa_on_tic_tac_toe_solo(tictactoe))
    # print(q_learning_on_tic_tac_toe_solo(tictactoe))
    # print(expected_sarsa_on_tic_tac_toe_solo(tictactoe))

    # print(sarsa_on_secret_env3())
    # print(q_learning_on_secret_env3())
    # print(expected_sarsa_on_secret_env3())
    print("End")
