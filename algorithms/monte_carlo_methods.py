from typing import Any

import numpy as np

from do_not_touch.result_structures import PolicyAndActionValueFunction
from do_not_touch.single_agent_env_wrapper import Env2
from envs.tictactoe import TicTacToe


def monte_carlo_es_on_tic_tac_toe_solo(
        env: Any = TicTacToe(), num_episodes: int = 10000, gamma: float = 0.1
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    pi, q, returns = {}, {}, {}

    for _ in range(num_episodes):
        env.reset_random()
        S, A, R = [], [], []

        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()

            if s not in pi:
                pi[s], q[s], returns[s] = {}, {}, {}
                for a in available_actions:
                    pi[s][a] = q[s][a] = returns[s][a] = 0 if s in pi else 1.0 / len(available_actions)

            A.append(np.random.choice(available_actions, 1, False)[0])
            old_score = env.score()
            env.act_with_action_id(A[-1])
            R.append(env.score() - old_score)
        G = 0

        for t in reversed(range(len(S))):
            G = G * gamma + R[t]
            if not any(S[t] == s and A[t] == a for s, a in zip(S[:t], A[:t])):
                q[S[t]][A[t]] = (q[S[t]][A[t]] * returns[S[t]][A[t]] + G) / (returns[S[t]][A[t]] + 1)
                returns[S[t]][A[t]] += 1
                pi[S[t]] = list(q[S[t]].keys())[np.argmax(list(q[S[t]].values()))]

    return PolicyAndActionValueFunction(pi, q)


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(
        env: Any = TicTacToe(), num_episodes: int = 10000, gamma: float = 0.1
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its Action-Value function (Q(s,a))
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    q, c_weights, pi, b_policy = {}, {}, {}, {}

    rng = np.random.default_rng()

    for _ in range(num_episodes):
        env.reset()
        S, A, R = [], [], []

        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            if s not in b_policy:
                available_actions = env.available_actions_ids()
                nb_of_actions = len(available_actions)
                proba_s = rng.integers(1, nb_of_actions, nb_of_actions, endpoint=True).astype(float)
                proba_s /= sum(proba_s)
                b_policy[s] = {a: proba_s[id_a] for id_a, a in enumerate(available_actions)}

            available_actions = env.available_actions_ids()

            if s not in q:
                q[s] = {a: 0.0 for a in available_actions}
                c_weights[s] = {a: 0.0 for a in available_actions}

            A.append(np.random.choice(list(b_policy[s].keys()), 1, False, p=list(b_policy[s].values()))[0])
            old_score = env.score()
            env.act_with_action_id(A[-1])
            R.append(env.score() - old_score)

        G, W = 0.0, 1.0

        for t in reversed(range(len(S))):
            s_t, a_t = S[t], A[t]
            G = G * gamma + R[t]
            c_weights[s_t][a_t] += W
            q[s_t][a_t] += (W / c_weights[s_t][a_t] * (G - q[s_t][a_t]))
            pi[s_t] = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]

            if a_t == pi[s_t]:
                break

            W = W / b_policy[s_t][a_t]

    return PolicyAndActionValueFunction(pi, q)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo(
        env: Any = TicTacToe(), num_episodes: int = 10000, gamma: float = 0.1, epsilon: float = 0.1
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    assert (epsilon > 0)
    pi, q, returns = {}, {}, {}

    for _ in range(num_episodes):
        env.reset_random()
        S, A, R = [], [], []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {a: 1.0 / len(available_actions) for a in available_actions}
                q[s] = {a: 0.0 for a in available_actions}
                returns[s] = {a: 0 for a in available_actions}

            A.append(np.random.choice(list(pi[s].keys()), 1, False, p=list(pi[s].values()))[0])
            old_score = env.score()
            env.act_with_action_id(A[-1])
            R.append(env.score() - old_score)

        G = 0
        for t in reversed(range(len(S))):
            G = G * gamma + R[t]
            s_t, a_t = S[t], A[t]
            if not any(s_t == s and a_t == a for s, a in zip(S[:t], A[:t])):
                q[s_t][a_t] = (q[s_t][a_t] * returns[s_t][a_t] + G) / (returns[s_t][a_t] + 1)
                returns[s_t][a_t] += 1
                optimal_s_a = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
                available_action_t_counts = len(q[s_t])
                for a in q[s_t]:
                    pi[s_t][a] = (1 - epsilon + epsilon / available_action_t_counts) \
                        if a == optimal_s_a \
                        else epsilon / available_action_t_counts
    return PolicyAndActionValueFunction(pi, q)


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    return monte_carlo_es_on_tic_tac_toe_solo(env=Env2(), num_episodes=10000, gamma=0.1)


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> (
        PolicyAndActionValueFunction
):
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    return on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(env=Env2(), num_episodes=10000, gamma=0.1)


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    return off_policy_monte_carlo_control_on_tic_tac_toe_solo(env=Env2(), num_episodes=10000, gamma=0.1, epsilon=0.1)


def demo():
    print("Monte Carlo ES on Tic Tac Toe Solo")
    print(monte_carlo_es_on_tic_tac_toe_solo())
    print("On Policy First Visit Monte Carlo Control on Tic Tac Toe Solo")
    print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    print("Off Policy Monte Carlo Control on Tic Tac Toe Solo")
    print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    print("Monte Carlo ES on Secret Env2")
    print(monte_carlo_es_on_secret_env2())
    print("On Policy First Visit Monte Carlo Control on Secret Env2")
    print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    print("Off Policy Monte Carlo Control on Secret Env2")
    print(off_policy_monte_carlo_control_on_secret_env2())


if __name__ == "__main__":
    demo()
