from do_not_touch.result_structures import PolicyAndActionValueFunction
from do_not_touch.single_agent_env_wrapper import Env3
from games.line_world import SingleAgentLineWorldEnv, LineWorldEnv
from games.grid_world import SingleAgentGridWorldEnv, GridWorldEnv
from algorithms.utils import return_policy_into_dict
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


def expected_sarsa_on_line_world() -> PolicyAndActionValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass

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

def expected_sarsa_on_grid_world() -> PolicyAndActionValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass

def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def q_learning_on_tic_tac_toe_solo(env,
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
    # TODO
    pass


def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def sarsa_on_secret_env1() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env1
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def q_learning_on_secret_env1(env,
                             gamma: float = 0.9999,
                             alpha: float = 0.01,
                             epsilon: float = 0.2,
                             max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env1
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def expected_sarsa_on_secret_env1() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env1
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass

def sarsa_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def q_learning_on_secret_env2(env,
                             gamma: float = 0.9999,
                             alpha: float = 0.01,
                             epsilon: float = 0.2,
                             max_episodes_count: int = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def expected_sarsa_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def q_learning_on_secret_env3(env,
                             gamma: float = 0.9999,
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
    # TODO
    pass


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def demo():
    line_world = LineWorldEnv(7)
    grid_world = GridWorldEnv(5, 5)
    print(sarsa_on_line_world(line_world))
    print(q_learning_on_line_world(line_world))
    #print(expected_sarsa_on_line_world())

    print(sarsa_on_grid_world(grid_world))
    print(q_learning_on_grid_world(grid_world))
    #print(expected_sarsa_on_grid_world())

    """
    print(sarsa_on_tic_tac_toe_solo())
    print(q_learning_on_tic_tac_toe_solo())
    print(expected_sarsa_on_tic_tac_toe_solo())

    print(sarsa_on_secret_env1())
    print(q_learning_on_secret_env1())
    print(expected_sarsa_on_secret_env1())

    print(sarsa_on_secret_env2())
    print(q_learning_on_secret_env2())
    print(expected_sarsa_on_secret_env2())

    print(sarsa_on_secret_env3())
    print(q_learning_on_secret_env3())
    print(expected_sarsa_on_secret_env3())
    """
