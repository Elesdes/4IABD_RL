import random

from games.tictactoe import TicTacToe

from do_not_touch.result_structures import PolicyAndActionValueFunction
from do_not_touch.single_agent_env_wrapper import Env2


def monte_carlo_es_on_tic_tac_toe_solo(
    num_episodes: int = 10000, epsilon: float = 0.1
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = TicTacToe()
    Q = {}
    N = {}
    pi = {}

    for _ in range(num_episodes):
        env.reset()
        state_actions = []

        while not env.is_game_over():
            state = env.board.copy()
            actions = env.available_actions()

            if len(actions) == 0:
                break

            # Generate true random values between 0 and 1
            action = (
                random.choice(actions)
                if random.uniform(0, 1) < epsilon
                else pi.get(tuple(state.tobytes()), random.choice(actions))
            )
            state_actions.append((state, action))
            env.play(action)

        G = 0 if not env.is_game_over() else 1 if env.player == -1 else -1

        for state, action in state_actions:
            key = (tuple(state.tobytes()), tuple(action))
            N[key] = N.get(key, 0) + 1
            Q[key] = Q.get(key, 0) + (G - Q.get(key, 0)) / N[key]

            available_actions = env.available_actions()
            if len(available_actions) > 0:
                pi[tuple(state.tobytes())] = max(
                    available_actions,
                    key=lambda a: Q.get((tuple(state.tobytes()), tuple(a)), 0),
                )

    return pi


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> (
    PolicyAndActionValueFunction
):
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> (
    PolicyAndActionValueFunction
):
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    # TODO


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> (
    PolicyAndActionValueFunction
):
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO


def demo():
    print(monte_carlo_es_on_tic_tac_toe_solo())
    print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    print(monte_carlo_es_on_secret_env2())
    print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    print(off_policy_monte_carlo_control_on_secret_env2())
