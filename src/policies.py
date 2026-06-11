"""
policies.py
Policy classes for CSCN8020 Assignment 1.
"""

import numpy as np


class Policy:
    """
    Wraps a deterministic or stochastic policy over a discrete action space.

    A deterministic policy stores action[s] for each state.
    A stochastic policy stores a probability distribution over actions for each state.

    Supports array-style indexing (policy[s]) so it integrates transparently
    with numpy operations and the existing visualisation utilities.
    """

    def __init__(self, n_states: int, n_actions: int, stochastic: bool = False):
        self.n_states   = n_states
        self.n_actions  = n_actions
        self.stochastic = stochastic

        if stochastic:
            self.probs   = np.ones((n_states, n_actions)) / n_actions
        else:
            self.actions = np.zeros(n_states, dtype=int)

    # ── Array-style access ────────────────────────────────────────────────────

    def __getitem__(self, key):
        return self.actions[key]

    def __setitem__(self, key, value):
        self.actions[key] = value

    def __array__(self, dtype=None):
        """Let numpy treat a Policy as its underlying actions array."""
        return self.actions if dtype is None else self.actions.astype(dtype)

    def __len__(self):
        return self.n_states

    def tolist(self):
        return self.actions.tolist()

    # ── Policy methods ────────────────────────────────────────────────────────

    def select_action(self, state: int) -> int:
        """Sample an action for the given state."""
        if self.stochastic:
            return int(np.random.choice(self.n_actions, p=self.probs[state]))
        return int(self.actions[state])

    def prob(self, state: int, action: int) -> float:
        """Return P(action | state)."""
        if self.stochastic:
            return float(self.probs[state, action])
        return 1.0 if self.actions[state] == action else 0.0

    def update_from_array(self, action_array: np.ndarray):
        """Set deterministic policy from an array of action indices."""
        if self.stochastic:
            raise ValueError("Use probs= for stochastic policies.")
        self.actions = action_array.copy()

    def make_greedy_from_V(self, env, V: np.ndarray):
        """
        Set deterministic policy to argmax_a Q(s,a) for all states.
        Uses the environment's reward_matrix and transitions tables.
        """
        if self.stochastic:
            raise ValueError("make_greedy_from_V produces a deterministic policy.")
        for s in range(self.n_states):
            if env.is_goal(s):
                self.actions[s] = 0
                continue
            Q = (env.reward_matrix[s]
                 + env.gamma * V[env.transitions[s]])
            self.actions[s] = int(np.argmax(Q))

    def __repr__(self):
        kind = "stochastic" if self.stochastic else "deterministic"
        return f"Policy({kind}, n_states={self.n_states}, n_actions={self.n_actions})"
