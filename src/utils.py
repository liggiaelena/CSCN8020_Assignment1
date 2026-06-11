"""
utils.py
Visualization and helper utilities for CSCN8020 Assignment 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def plot_gridworld(env, V, policy, title: str = "", save: bool = True):
    """
    Draw two side-by-side heatmaps: V* values and pi* arrows.

    Parameters
    ----------
    env    : GridWorld5x5 instance
    V      : value function array, shape (n_states,)
    policy : policy array of action indices, shape (n_states,)
    title  : figure title suffix
    save   : whether to save the figure to images/
    """
    Path("images").mkdir(exist_ok=True)

    rows, cols = env.ROWS, env.COLS
    fig, axes  = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"5×5 Gridworld — {title}", fontsize=14, fontweight="bold")

    # ── Build color matrix ────────────────────────────────────────────────────
    # Normalise V for the colour map (ignoring terminal which is 0)
    non_terminal = [V[s] for s in range(env.n_states) if not env.is_goal(s)]
    v_min, v_max = min(non_terminal), max(non_terminal)

    color_matrix = np.zeros((rows, cols, 3))
    for s in range(env.n_states):
        r, c = env.to_pos(s)
        if env.is_goal(s):
            color_matrix[r, c] = [0.18, 0.72, 0.18]   # green
        elif env.is_grey(s):
            color_matrix[r, c] = [0.55, 0.55, 0.55]   # grey
        else:
            # Cool-warm gradient based on value
            t = (V[s] - v_min) / (v_max - v_min + 1e-9)
            color_matrix[r, c] = [0.6 + 0.3 * t, 0.8, 1.0 - 0.4 * t]

    for ax_idx, ax in enumerate(axes):
        ax.imshow(color_matrix, aspect="equal", interpolation="nearest")

        for s in range(env.n_states):
            r, c  = env.to_pos(s)
            txt_c = "white" if env.is_grey(s) or env.is_goal(s) else "black"

            if ax_idx == 0:
                # Value function panel
                label = "GOAL" if env.is_goal(s) else f"{V[s]:.2f}"
                ax.text(c, r, label, ha="center", va="center",
                        fontsize=8.5, color=txt_c, fontweight="bold")
            else:
                # Policy panel
                label = "G" if env.is_goal(s) else env.ACTION_ARROWS[policy[s]]
                ax.text(c, r, label, ha="center", va="center",
                        fontsize=14, color=txt_c)

        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticklabels([f"c{i}" for i in range(cols)])
        ax.set_yticklabels([f"r{i}" for i in range(rows)])
        ax.set_title("V* (Optimal Values)" if ax_idx == 0 else "π* (Optimal Policy)",
                     fontsize=12)
        ax.tick_params(length=0)

        # Grid lines
        for x in np.arange(-0.5, cols, 1):
            ax.axvline(x, color="white", linewidth=1.5)
        for y in np.arange(-0.5, rows, 1):
            ax.axhline(y, color="white", linewidth=1.5)

    plt.tight_layout()
    if save:
        fname = f"images/{title.replace(' ', '_').replace('*', '')}.png"
        plt.savefig(fname, dpi=130, bbox_inches="tight")
        print(f"Figure saved: {fname}")
    plt.show()


def print_value_table(env, V, label: str = "V"):
    """Print V as a formatted rows x cols table."""
    print(f"\n  {label} (row × col):")
    for r in range(env.ROWS):
        row_str = "  "
        for c in range(env.COLS):
            s = env.to_idx(r, c)
            row_str += f"{V[s]:7.3f} "
        print(row_str)
    print()


def print_policy_table(env, policy):
    """Print policy as arrows in a rows x cols table."""
    print("\n  π* (row × col):")
    for r in range(env.ROWS):
        row_str = "  "
        for c in range(env.COLS):
            s = env.to_idx(r, c)
            a = "G" if env.is_goal(s) else env.ACTION_ARROWS[policy[s]]
            row_str += f"{a:>4} "
        print(row_str)
    print()
