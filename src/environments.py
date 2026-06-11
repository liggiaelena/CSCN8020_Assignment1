"""
environments.py
Gridworld environment classes for CSCN8020 Assignment 1.
"""

import numpy as np


class GridWorld5x5:
    """
    5x5 episodic gridworld MDP (Problems 3 and 4).

    Reward is received upon ENTERING a state:
        Goal  (4,4)              : +10
        Grey  {(2,2),(3,0),(0,4)}: -5
        All other states         : -1

    V*(goal) = 0  (terminal — no future rewards after reaching goal).

    Actions : 0=up, 1=down, 2=left, 3=right
    Transitions : deterministic; wall actions leave agent in place.
    """

    ROWS, COLS = 5, 5
    GOAL        = (4, 4)
    GREY_STATES = frozenset({(2, 2), (3, 0), (0, 4)})

    ACTION_NAMES  = ["up", "down", "left", "right"]
    ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_ARROWS = ["↑", "↓", "←", "→"]   # ↑ ↓ ← →

    R_GOAL    = +10.0
    R_GREY    =  -5.0
    R_REGULAR =  -1.0

    def __init__(self, gamma: float = 0.9):
        self.gamma     = gamma
        self.n_states  = self.ROWS * self.COLS
        self.n_actions = len(self.ACTION_DELTAS)
        self.goal_idx  = self.to_idx(*self.GOAL)

        self.transitions   = self._build_transitions()    # shape (n_states, n_actions)
        self.entry_rewards = self._build_entry_rewards()  # shape (n_states,)
        # reward_matrix[s, a] = R(T(s,a))
        self.reward_matrix = self.entry_rewards[self.transitions]

    # ── Index helpers ──────────────────────────────────────────────────────────

    def to_idx(self, row: int, col: int) -> int:
        return row * self.COLS + col

    def to_pos(self, idx: int):
        return (idx // self.COLS, idx % self.COLS)

    def is_goal(self, idx: int) -> bool:
        return idx == self.goal_idx

    def is_grey(self, idx: int) -> bool:
        return self.to_pos(idx) in self.GREY_STATES

    # ── Build tables ──────────────────────────────────────────────────────────

    def _build_entry_rewards(self) -> np.ndarray:
        R = np.full(self.n_states, self.R_REGULAR)
        for (r, c) in self.GREY_STATES:
            R[self.to_idx(r, c)] = self.R_GREY
        R[self.goal_idx] = self.R_GOAL
        return R

    def _build_transitions(self) -> np.ndarray:
        T = np.zeros((self.n_states, self.n_actions), dtype=int)
        for s in range(self.n_states):
            r, c = self.to_pos(s)
            for a, (dr, dc) in enumerate(self.ACTION_DELTAS):
                nr, nc = r + dr, c + dc
                T[s, a] = (
                    self.to_idx(nr, nc)
                    if 0 <= nr < self.ROWS and 0 <= nc < self.COLS
                    else s
                )
        return T
