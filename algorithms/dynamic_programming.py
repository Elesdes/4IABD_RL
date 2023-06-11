from do_not_touch.mdp_env_wrapper import Env1
from do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
import numpy as np
from algorithms.utils import return_policy_into_dict
from games.line_world import LineWorldEnv, SingleAgentLineWorldEnv
from games.grid_world import GridWorldEnv, SingleAgentGridWorldEnv
from do_not_touch.mdp_env_wrapper import Env1
from do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction, Policy


def policy_evaluation_on_line_world(env: SingleAgentLineWorldEnv,
                                    theta: float = 0.0000001) -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    V = np.zeros((env.state_space(),))
    while True:
        delta = 0.0
        for s in range(env.state_space()):
            old_v = V[s]
            total = 0.0
            for a in env.available_actions():
                total_inter = 0.0
                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total_inter += env.p(s, a, s_p, r) * (env.possible_score()[r] + 0.99999 * V[s_p])
                total_inter = env.pi(s, a) * total_inter
                total += total_inter
            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))
        if delta < theta:
            return dict(enumerate(V.flatten(), 1))


def policy_iteration_on_line_world(env: SingleAgentLineWorldEnv,
                                   theta: float = 0.0000001) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    V = np.zeros((env.state_space(),))
    pi = np.random.randint(0, len(env.available_actions()), env.state_space())

    while True:
        # policy evaluation
        while True:
            delta = 0
            for s in range(env.state_space()):
                old_v = V[s]

                total = 0.0

                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total += env.p(s, pi[s], s_p, r) * (env.possible_score()[r] + 0.99999999 * V[s_p])
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
                        total += env.p(s, a, s_p, r) * (env.possible_score()[r] + 0.999999 * V[s_p])
                if best_a is None or total > best_action_score:
                    best_a = a
                    best_action_score = total

            pi[s] = best_a
            if best_a != old_action:
                policy_stable = False

        if policy_stable:
            return PolicyAndValueFunction(pi=dict(enumerate(pi)), v=dict(enumerate(V.flatten(), 1)))


def value_iteration_on_line_world(env: SingleAgentLineWorldEnv,
                                  theta: float = 0.0000001) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    V = np.zeros((env.state_space(),))
    pi = np.random.randint(0, len(env.available_actions()), env.state_space())
    while True:
        delta = 0
        for s in range(env.state_space()):
            old_v = V[s]

            best_action_score = None
            best_a = None
            for a in env.available_actions():
                total = 0.0
                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total += env.p(s, a, s_p, r) * (env.possible_score()[r] + 0.99999999 * V[s_p])

                if best_action_score is None or total > best_action_score:
                    best_action_score = total
                    best_a = a

            V[s] = best_action_score
            pi[s] = best_a
            delta = max(delta, np.abs(old_v - V[s]))
        if delta < theta:
            break

    return PolicyAndValueFunction(pi=dict(enumerate(pi)), v=dict(enumerate(V.flatten(), 1)))

def policy_evaluation_on_grid_world(env: SingleAgentGridWorldEnv,
                                    theta: float = 0.0000001) -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    V = np.zeros((env.state_space(),))
    while True:
        delta = 0.0
        for s in range(env.state_space()):
            old_v = V[s]
            total = 0.0
            for a in env.available_actions():
                total_inter = 0.0
                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total_inter += env.p(s, a, s_p, r) * (env.possible_score()[r] + 0.99999 * V[s_p])
                total_inter = env.pi(s, a) * total_inter
                total += total_inter
            V[s] = total
            delta = max(delta, np.abs(V[s] - old_v))
        if delta < theta:
            return dict(enumerate(V.flatten(), 1))


def policy_iteration_on_grid_world(env: SingleAgentGridWorldEnv,
                                   theta: float = 0.0000001) -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    V = np.zeros((env.state_space(),))
    pi = np.random.randint(0, len(env.available_actions()), env.state_space())

    while True:
        # policy evaluation
        while True:
            delta = 0
            for s in range(env.state_space()):
                old_v = V[s]

                total = 0.0

                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total += env.p(s, pi[s], s_p, r) * (env.possible_score()[r] + 0.99999999 * V[s_p])
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
                        total += env.p(s, a, s_p, r) * (env.possible_score()[r] + 0.999999 * V[s_p])
                if best_a is None or total > best_action_score:
                    best_a = a
                    best_action_score = total

            pi[s] = best_a
            if best_a != old_action:
                policy_stable = False

        if policy_stable:
            return PolicyAndValueFunction(pi=dict(enumerate(pi)), v=dict(enumerate(V.flatten(), 1)))


def value_iteration_on_grid_world(env: SingleAgentGridWorldEnv,
                                  theta: float = 0.0000001) -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    V = np.zeros((env.state_space(),))
    pi = np.random.randint(0, len(env.available_actions()), env.state_space())
    while True:
        delta = 0
        for s in range(env.state_space()):
            old_v = V[s]

            best_action_score = None
            best_a = None
            for a in env.available_actions():
                total = 0.0
                for s_p in range(env.state_space()):
                    for r in range(len(env.possible_score())):
                        total += env.p(s, a, s_p, r) * (env.possible_score()[r] + 0.99999999 * V[s_p])

                if best_action_score is None or total > best_action_score:
                    best_action_score = total
                    best_a = a

            V[s] = best_action_score
            pi[s] = best_a
            delta = max(delta, np.abs(old_v - V[s]))
        if delta < theta:
            break

    return PolicyAndValueFunction(pi=dict(enumerate(pi)), v=dict(enumerate(V.flatten(), 1)))


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    # TODO
    pass


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def demo():
    line_world = LineWorldEnv(7)
    grid_world = GridWorldEnv(5,5)
    print(policy_evaluation_on_line_world(line_world))
    print(policy_iteration_on_line_world(line_world))
    print(value_iteration_on_line_world(line_world))

    print(policy_evaluation_on_grid_world(grid_world))
    print(policy_iteration_on_grid_world(grid_world))
    print(value_iteration_on_grid_world(grid_world))
    
    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())