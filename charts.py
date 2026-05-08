
from typing import Dict, List, Tuple
from helpers.viz import evaluate
import numpy as np
import matplotlib.pyplot as plt
from ModifiedSlipperyGridWorld import ModifiedSlipperyGridWorld
from algorithms import ValueIteration, QLearning, SARSA, DynaQ


def stability_study(
        algorithms: Dict[str, callable],
        env_config: dict,
        algo_params: Dict[str, dict],
        seeds: List[int],
        eval_episodes: int = 100
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run multiple algorithms across multiple seeds and collect metrics.

    Args:
        algorithms: {"QLearning": QLearning, "SARSA": SARSA, ...}
        env_config: {rows, cols, start, goal, slip_prob, ...} without seed
        algo_params: {"QLearning": {epsilon, gamma, ...}, "SARSA": {...}, ...}
        seeds: List of seeds to test
        eval_episodes: Episodes for evaluation

    Returns:
        {algo_name: {"metric": array_of_values}}
    """
    metrics = ['avg_return', 'std_return', 'success_rate', 'trap_rate', 'avg_steps']
    results = {name: {m: [] for m in metrics} for name in algorithms}

    for seed in seeds:
        env_config['seed'] = seed
        env = ModifiedSlipperyGridWorld(**env_config)

        for name, algo_func in algorithms.items():
            result = algo_func(env, **algo_params[name])
            V, pi = result[0], result[1]
            stats = evaluate(env, policy=pi, n_episodes=eval_episodes, seed=seed)

            for m in metrics:
                results[name][m].append(stats[m])

    for name in results:
        for m in metrics:
            results[name][m] = np.array(results[name][m])

    return results


# def plot_stability(
#         results: Dict[str, Dict[str, np.ndarray]],
#         seeds: List[int],
#         figsize: Tuple[int, int] = (15, 10)
# ):
#     """
#     Plot stability results: boxplots + line plots per metric.
#     """
#     metrics_labels = {
#         'avg_return': 'Average Return',
#         'std_return': 'Std of Return',
#         'success_rate': 'Success Rate',
#         'trap_rate': 'Trap Rate',
#         'avg_steps': 'Average Steps'
#     }
#
#     algo_names = list(results.keys())
#     colors = plt.cm.tab10(np.linspace(0, 1, len(algo_names)))
#
#     fig, axes = plt.subplots(2, 3, figsize=figsize)
#     axes = axes.flatten()
#
#     for ax, (metric, label) in zip(axes, metrics_labels.items()):
#         data = [results[name][metric] for name in algo_names]
#         bp = ax.boxplot(data, labels=algo_names, patch_artist=True,
#                         showmeans=True,
#                         meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markersize': 8})
#
#         for patch, color in zip(bp['boxes'], colors):
#             patch.set_facecolor(color)
#             patch.set_alpha(0.7)
#
#         for i, (name, d) in enumerate(zip(algo_names, data)):
#             x_jitter = np.random.normal(i + 1, 0.04, size=len(d))
#             ax.scatter(x_jitter, d, alpha=0.5, color='black', s=15, zorder=3)
#
#         ax.set_title(label, fontweight='bold')
#         ax.grid(axis='y', alpha=0.3)
#
#     if len(metrics_labels) < len(axes):
#         axes[-1].remove()
#
#     plt.suptitle('Algorithm Stability Analysis', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.show()
#
#     fig2, axes2 = plt.subplots(2, 3, figsize=figsize)
#     axes2 = axes2.flatten()
#
#     for ax, (metric, label) in zip(axes2, metrics_labels.items()):
#         for name, color in zip(algo_names, colors):
#             mean_vals = results[name][metric]
#             ax.plot(seeds, mean_vals, 'o-', color=color, label=name,
#                     linewidth=2, markersize=6)
#
#         ax.set_title(label, fontweight='bold')
#         ax.set_xlabel('Seed')
#         ax.set_ylabel(label)
#         ax.legend(fontsize=8)
#         ax.grid(alpha=0.3)
#
#     if len(metrics_labels) < len(axes2):
#         axes2[-1].remove()
#
#     plt.suptitle('Stability Across Seeds', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.show()
def plot_stability(
        results: Dict[str, Dict[str, np.ndarray]],
        seeds: List[int]
):
    """
    Plot stability results - each metric as a separate figure.
    Figure 1: Boxplots comparing algorithms
    Figure 2: Scatter plots showing variation across seeds
    """
    metrics_labels = {
        'avg_return': 'Average Return',
        'std_return': 'Std of Return',
        'success_rate': 'Success Rate',
        'trap_rate': 'Trap Rate',
        'avg_steps': 'Average Steps'
    }

    algo_names = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(algo_names)))

    # ============================================
    # FIGURE 1: Boxplots for each metric
    # ============================================
    for metric, label in metrics_labels.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        data = [results[name][metric] for name in algo_names]
        bp = ax.boxplot(data, labels=algo_names, patch_artist=True,
                        showmeans=True,
                        meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markersize': 8})

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Individual seed points
        for i, (name, d) in enumerate(zip(algo_names, data)):
            x_jitter = np.random.normal(i + 1, 0.04, size=len(d))
            ax.scatter(x_jitter, d, alpha=0.5, color='black', s=20, zorder=3)

        ax.set_title(f'{label} - Distribution Across Seeds', fontweight='bold', fontsize=14)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Add mean ± std annotation
        for i, (name, d) in enumerate(zip(algo_names, data)):
            mean_val = np.mean(d)
            std_val = np.std(d)
            ax.annotate(f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                        xy=(i + 1, mean_val),
                        xytext=(i + 1.3, ax.get_ylim()[1] * 0.95),
                        fontsize=8,
                        color='darkred',
                        ha='center')

        plt.tight_layout()
        plt.show()

    # ============================================
    # FIGURE 2: Scatter plots per seed
    # ============================================
    for metric, label in metrics_labels.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for name, color in zip(algo_names, colors):
            values = results[name][metric]
            ax.scatter(seeds, values, color=color, label=name,
                       s=80, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)

            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.5, linewidth=1.5)

            # Add std band
            std_val = np.std(values)
            ax.fill_between([min(seeds) - 0.5, max(seeds) + 0.5],
                            mean_val - std_val,
                            mean_val + std_val,
                            alpha=0.1, color=color)

        ax.set_title(f'{label} - Variation Across Seeds', fontweight='bold', fontsize=14)
        ax.set_xlabel('Seed', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)

        # Set x-ticks to actual seed values
        ax.set_xticks(seeds)
        ax.set_xticklabels(seeds, rotation=45, fontsize=8)

        plt.tight_layout()
        plt.show()