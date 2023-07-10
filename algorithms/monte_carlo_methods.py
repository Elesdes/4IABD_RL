import numpy as np

from do_not_touch.result_structures import PolicyAndActionValueFunction
from do_not_touch.single_agent_env_wrapper import Env2
from games.tictactoe import TicTacToe


def monte_carlo_es_on_tic_tac_toe_solo(
        num_episodes: int = 10000, epsilon: float = 0.1
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = TicTacToe()
    q = {}
    visit_counts = {}
    for _ in range(num_episodes):
        env.reset()
        # Random initial action for exploring start
        initial_action = tuple(env.available_actions()[np.random.randint(len(env.available_actions()))])
        env.play(initial_action)
        episode_history = [(tuple(env.board.flatten()), initial_action)]
        while not env.is_game_over():
            state = tuple(env.board.flatten())
            actions = env.available_actions()

            if state not in q:
                q[state] = {tuple(action): 0 for action in actions}
                visit_counts[state] = {tuple(action): 0 for action in actions}

            action = (
                actions[np.random.randint(len(actions))]
                if np.random.rand() < epsilon
                else max(visit_counts[state], key=visit_counts[state].get)
            )
            env.play(action)
            episode_history.append((state, action))

        reward = 1 if env.player == -1 else -1
        for state, action in episode_history:
            if action not in visit_counts[state]:
                visit_counts[state][action] = 0

            visit_counts[state][action] += 1
            q[state][action] += (reward - q[state][action]) / visit_counts[state][action]
            reward = -reward

    pi = {state: max(q[state], key=q[state].get) for state in q}
    return PolicyAndActionValueFunction(pi=pi, q=q)


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(
        num_episodes: int = 10000, epsilon: float = 0.1
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()

    # Initialize the action-value function Q(s, a) and the policy Pi(s, a) for all states and actions
    Q = {}
    N = np.zeros((3, 3, 2, 2))
    Pi = {}

    for _ in range(num_episodes):
        episode = []
        state = env.reset()
        done = False

        # Generate an episode using the current policy
        while not done:
            if state not in Pi:
                Pi[state] = {action: 0.5 for action in env.available_actions()}

            action = (
                np.random.choice(env.available_actions())
                if np.random.rand() < epsilon
                else max(Pi[state], key=Pi[state].get)
            )
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Update the action-value function and the policy
        G = 0
        for s, a, r in reversed(episode):
            G = r + G
            if s not in Q:
                Q[s] = {action: 0 for action in env.available_actions()}
            if s not in N:
                N[s] = {action: 0 for action in env.available_actions()}

            N[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / N[s][a]
            Pi[s] = {action: 0 for action in env.available_actions()}
            Pi[s][a] = 1 - epsilon + epsilon / len(env.available_actions())

    return PolicyAndActionValueFunction(pi=Pi, q=Q)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo(
        num_episodes: int = 10000, epsilon: float = 0.1
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    num_actions = 9
    Q = {}
    Nsa = {}
    Pi = {}

    for _ in range(num_episodes):
        episode = []
        state = 0
        env.reset()

        # Generate episode using behavior policy
        while not env.is_game_over():
            if state not in Pi:
                Pi[state] = {action: 1 / num_actions for action in range(num_actions)}

            action = np.random.choice(num_actions, p=list(Pi[state].values()))
            episode.append((state, action))
            state, _ = env.play((action // 3, action % 3))

        # Calculate returns and update Q-values and target policy
        G = 0
        W = 1
        for t in range(len(episode) - 1, -1, -1):
            state, action = episode[t]
            reward = env.board.flatten()[state // 3 ** (8 - action)]
            G = reward + G

            if state not in Nsa:
                Nsa[state] = {action: 0 for action in range(num_actions)}

            Nsa[state][action] += W
            alpha = W / Nsa[state][action]

            if state not in Q:
                Q[state] = {action: 0 for action in range(num_actions)}

            Q[state][action] += alpha * (G - Q[state][action])

            a_star = max(Pi[state], key=Pi[state].get)
            for a in Pi[state]:
                Pi[state][a] = 1 - epsilon + epsilon / num_actions if a == a_star else epsilon / num_actions

            if action != a_star:
                break
            W = W / epsilon if action == a_star else 0

    return PolicyAndActionValueFunction(pi=Pi, q=Q)


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()


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


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()


def demo():
    print(monte_carlo_es_on_tic_tac_toe_solo())
    print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    print(monte_carlo_es_on_secret_env2())
    print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    print(off_policy_monte_carlo_control_on_secret_env2())


if __name__ == "__main__":
    demo()