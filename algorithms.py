import numpy as np
import random
from helpers.env import ACTIONS
from ModifiedSlipperyGridWorld import ModifiedSlipperyGridWorld
from typing import Tuple
from helpers.viz import greedy_policy_from_V

def ValueIteration(env: ModifiedSlipperyGridWorld, max_number_iterations:int, gamma:float, threshold:float) -> Tuple[np.ndarray, np.ndarray]:
    env.reset()
    num_states = env.num_states
    V = np.zeros(num_states)
    for i in range(max_number_iterations):
        V_i = V.copy()
        d = 0
        for s in range(num_states):
            if env.is_terminal_state(s):
                continue
            R = []
            for a in ACTIONS:
                val = 0
                for p, sn in env.get_transition_distribution(s, a):
                    r = env.reward(s, a, sn)
                    val += p * (r + gamma * V[sn])
                R.append(val)
            V_i[s] = max(R)
            d = max(d, abs(V_i[s] - V[s]))

        V = V_i
        if d < threshold:
            print(f"Iterations count: {i}")
            break

    pi = greedy_policy_from_V(V, env, gamma)
    return V, pi

def QLearning(env: ModifiedSlipperyGridWorld, epsilon:float, gamma:float, alpha: float, max_iterations: int) -> Tuple[np.ndarray, np.ndarray, list]:
    env.reset()
    num_states = env.num_states
    num_actions = 4
    num_elems = (num_states, num_actions)

    Q = np.zeros(num_elems)

    episode_returns = []

    def eps_greedy(state):
        if random.random() < epsilon:
            return random.randint(0, 3)
        return np.argmax(Q[state])

    for i in range(max_iterations):
        X = env.reset()
        done = False
        total_rewards = 0

        while not done:
            A = eps_greedy(X)
            X_prim, R, done, _ = env.step(A)

            Q[X][A] += + alpha * (R + gamma *(1-done)* np.max(Q[X_prim]) - Q[X][A])
            X = X_prim
            total_rewards += R

        episode_returns.append(total_rewards)

    def get_states(state):
        return np.max(Q[state])

    def greedy_policy(state):
        return int(np.argmax(Q[state]))

    V = np.zeros(num_states)
    for i in range(num_states):
        V[i] = get_states(i)

    pi = np.zeros(num_states)
    for i in range(num_states):
        pi[i] = greedy_policy(i)

    return V, pi, episode_returns

def SARSA(env: ModifiedSlipperyGridWorld, epsilon:float, gamma:float, alpha: float, max_iterations: int) -> Tuple[np.ndarray, np.ndarray, list]:
    env.reset()
    num_states = env.num_states
    num_actions = 4
    num_elems = (num_states, num_actions)
    Q = np.zeros(num_elems)

    episode_returns = []

    def eps_greedy(state):
        if random.random() < epsilon:
            return random.randint(0, 3)
        return np.argmax(Q[state])

    for i in range(max_iterations):
        X = env.reset()
        A = eps_greedy(X)
        done = False

        total_rewards = 0

        while not done:
            X_prim, R, done, _ = env.step(A)
            if done:
                Q[X][A] += alpha * (R - Q[X][A])
                total_rewards += R
                break

            A_prim = eps_greedy(X_prim)
            Q[X][A] = Q[X][A] + alpha * (R + gamma * Q[X_prim][A_prim] - Q[X][A])
            X = X_prim
            A = A_prim
            total_rewards += R

        episode_returns.append(total_rewards)

    def get_states(state):
        return np.max(Q[state])

    def get_policy(state):
        return int(np.argmax(Q[state]))

    V = np.zeros(num_states)
    for i in range(num_states):
        V[i] = get_states(i)

    pi = np.zeros(num_states)
    for i in range(num_states):
        pi[i] = get_policy(i)

    return V, pi, episode_returns

def DynaQ(env: ModifiedSlipperyGridWorld, epsilon:float, gamma:float, alpha: float, max_iterations: int, n: int) -> Tuple[np.ndarray, np.ndarray, list]:
    env.reset()
    num_states = env.num_states
    num_actions = 4
    num_elems = (num_states, num_actions)
    Q = np.zeros(num_elems)
    model = {}

    episode_returns = []

    def eps_greedy(state):
        if random.random() < epsilon:
            return random.randint(0, 3)
        return np.argmax(Q[state])

    for i in range(max_iterations):
        X = env.reset()
        done = False

        total_rewards = 0

        while not done:
            A = eps_greedy(X)
            X_prim, R, done, _ = env.step(A)

            tmp = R
            if not done:
                tmp += gamma * np.max(Q[X_prim])

            Q[X][A] += alpha * (tmp - Q[X][A])

            model[(X, A)] = (R, X_prim, done)
            model_keys = list(model.keys())

            for _ in range(min(n, len(model_keys))):
                prev_X, prev_A = random.choice(model_keys)
                prev_R, prev_X_prim, prev_done = model[(prev_X, prev_A)]

                tmp = prev_R
                if not prev_done:
                    tmp += gamma * np.max(Q[prev_X_prim])

                Q[prev_X, prev_A] += alpha * (tmp - Q[prev_X, prev_A])

            X = X_prim
            total_rewards += R

        episode_returns.append(total_rewards)

    def get_states(state):
        return np.max(Q[state])

    def greedy_policy(state):
        return int(np.argmax(Q[state]))

    V = np.zeros(num_states)
    for i in range(num_states):
        V[i] = get_states(i)

    pi = np.zeros(num_states)
    for i in range(num_states):
        pi[i] = greedy_policy(i)

    return V, pi, episode_returns
