import numpy as np
import random
import matplotlib.pyplot as plt

from ModifiedSlipperyGridWorld import ModifiedSlipperyGridWorld
from algorithms import QLearning, SARSA, DynaQ


def moving_average(values, window=100):
    values = np.array(values, dtype=float)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")

def run_learning_curve_for_seeds(env_config, algo_params, seeds):
    all_curves = {
        "Q-Learning": [],
        "SARSA": [],
        "Dyna-Q": []
    }

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        env_q = ModifiedSlipperyGridWorld(**env_config)
        _, _, q_returns = QLearning(
            env_q,
            **algo_params["Q-Learning"]
        )
        all_curves["Q-Learning"].append(q_returns)

        np.random.seed(seed)
        random.seed(seed)

        env_s = ModifiedSlipperyGridWorld(**env_config)
        _, _, s_returns = SARSA(
            env_s,
            **algo_params["SARSA"]
        )
        all_curves["SARSA"].append(s_returns)

        np.random.seed(seed)
        random.seed(seed)

        env_d = ModifiedSlipperyGridWorld(**env_config)
        _, _, d_returns = DynaQ(
            env_d,
            **algo_params["Dyna-Q"]
        )
        all_curves["Dyna-Q"].append(d_returns)

    return all_curves

def plot_learning_curves_mean(all_curves, window=100):
    plt.figure(figsize=(11, 6), dpi=140)

    for algo_name, curves in all_curves.items():
        smoothed_curves = []

        for curve in curves:
            smoothed_curve = moving_average(curve, window=window)
            smoothed_curves.append(smoothed_curve)

        min_len = min(len(curve) for curve in smoothed_curves)

        curves_array = np.array([
            curve[:min_len] for curve in smoothed_curves
        ])

        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)

        if len(curves[0]) >= window:
            episodes = np.arange(window - 1, window - 1 + min_len)
        else:
            episodes = np.arange(min_len)

        plt.plot(
            episodes,
            mean_curve,
            linewidth=2,
            label=algo_name
        )

        plt.fill_between(
            episodes,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15
        )

    plt.title("Learning Curve (Mean for different seeds)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()