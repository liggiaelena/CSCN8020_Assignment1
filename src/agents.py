"""
agents.py
RL agent classes for CSCN8020 Assignment 1.
"""

import numpy as np
import time
import logging

logger = logging.getLogger("RL_Assignment1")


class ValueIterationAgent:
    """
    Standard (synchronous) Value Iteration — Sutton & Barto (2018), p. 83.

    Two arrays are used: a frozen V_old and a V_new computed from it.
    All state updates in one sweep use the SAME (old) values.
    """

    def __init__(self, env, theta: float = 1e-6):
        self.env    = env
        self.theta  = theta
        self.V      = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int)

    def run(self):
        n_iter = 0
        t0     = time.perf_counter()

        while True:
            V_new = np.zeros(self.env.n_states)
            for s in range(self.env.n_states):
                if self.env.is_goal(s):
                    V_new[s] = 0.0   # terminal state fixed at 0
                    continue
                # Q(s,a) = R_entry(T(s,a)) + gamma * V_old(T(s,a))
                Q = (self.env.reward_matrix[s]
                     + self.env.gamma * self.V[self.env.transitions[s]])
                V_new[s] = float(np.max(Q))

            delta   = float(np.max(np.abs(V_new - self.V)))
            self.V  = V_new
            n_iter += 1

            if n_iter % 50 == 0:
                logger.info("[VI]    iter=%4d  delta=%.8f", n_iter, delta)
            if delta < self.theta:
                break

        elapsed = time.perf_counter() - t0
        self._extract_policy()
        logger.info("[VI]    Converged: iter=%d  time=%.4fs  delta=%.2e",
                    n_iter, elapsed, delta)
        return n_iter, elapsed

    def _extract_policy(self):
        for s in range(self.env.n_states):
            if self.env.is_goal(s):
                self.policy[s] = 0
                continue
            Q = (self.env.reward_matrix[s]
                 + self.env.gamma * self.V[self.env.transitions[s]])
            self.policy[s] = int(np.argmax(Q))


class InPlaceValueIterationAgent:
    """
    In-Place Value Iteration.

    Single array V updated immediately — updates within a sweep are visible
    to subsequent state updates in the SAME sweep.
    Typically converges in fewer iterations than standard VI.
    """

    def __init__(self, env, theta: float = 1e-6):
        self.env    = env
        self.theta  = theta
        self.V      = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int)

    def run(self):
        n_iter = 0
        t0     = time.perf_counter()

        while True:
            delta = 0.0
            for s in range(self.env.n_states):
                if self.env.is_goal(s):
                    continue   # terminal: never updated
                v_old     = self.V[s]
                # self.V may already contain updated values from this sweep
                Q         = (self.env.reward_matrix[s]
                             + self.env.gamma * self.V[self.env.transitions[s]])
                self.V[s] = float(np.max(Q))
                delta     = max(delta, abs(v_old - self.V[s]))
            n_iter += 1

            if n_iter % 50 == 0:
                logger.info("[IP-VI] iter=%4d  delta=%.8f", n_iter, delta)
            if delta < self.theta:
                break

        elapsed = time.perf_counter() - t0
        self._extract_policy()
        logger.info("[IP-VI] Converged: iter=%d  time=%.4fs  delta=%.2e",
                    n_iter, elapsed, delta)
        return n_iter, elapsed

    def _extract_policy(self):
        for s in range(self.env.n_states):
            if self.env.is_goal(s):
                self.policy[s] = 0
                continue
            Q = (self.env.reward_matrix[s]
                 + self.env.gamma * self.V[self.env.transitions[s]])
            self.policy[s] = int(np.argmax(Q))


class MonteCarloOffPolicyAgent:
    """
    Off-Policy Monte Carlo Control with Weighted Importance Sampling.
    Reference: Sutton & Barto (2018), p. 110.

    Behavior policy b : uniform random  b(a|s) = 1/|A|  (ensures coverage)
    Target  policy pi : greedy w.r.t. Q (updated after each backward pass)

    Reward convention: received upon ENTERING the next state (same as Problem 3).
    V*(s) = max_a Q*(s,a) extracted after training.
    V*(goal) = 0 (terminal).
    """

    def __init__(self, env, gamma: float = 0.9,
                 n_episodes: int = 50_000,
                 max_steps:  int = 500,
                 seed:       int = 42):
        self.env        = env
        self.gamma      = gamma
        self.n_episodes = n_episodes
        self.max_steps  = max_steps
        np.random.seed(seed)

        # Q[s, a] — action-value estimates, initialised to 0
        self.Q = np.zeros((env.n_states, env.n_actions))
        # C[s, a] — cumulative importance-sampling weights
        self.C = np.zeros((env.n_states, env.n_actions))

        # Target policy: greedy w.r.t. Q (initially all action-0)
        self.policy = np.zeros(env.n_states, dtype=int)

        # Behavior policy probability per action (uniform)
        self.b_prob = 1.0 / env.n_actions   # = 0.25

        self.V = None   # filled in by run()

    def _generate_episode(self) -> list:
        """
        Sample one episode under the BEHAVIOR policy (uniform random).
        Returns list of (state, action, reward) tuples.
        Reward = R_entry(next_state).
        """
        s = np.random.randint(0, self.env.n_states)
        while self.env.is_goal(s):
            s = np.random.randint(0, self.env.n_states)

        episode = []
        for _ in range(self.max_steps):
            a      = np.random.randint(0, self.env.n_actions)
            next_s = self.env.transitions[s, a]
            r      = self.env.entry_rewards[next_s]
            episode.append((s, a, r))
            s = next_s
            if self.env.is_goal(s):
                break
        return episode

    def run(self):
        """
        Run off-policy MC control for self.n_episodes episodes.
        Returns (n_episodes, elapsed_seconds).
        """
        t0 = time.perf_counter()

        for ep in range(self.n_episodes):
            episode = self._generate_episode()
            G = 0.0   # accumulated return (backward)
            W = 1.0   # cumulative IS weight

            # Backward pass: t = T-1, T-2, ..., 0
            for t in range(len(episode) - 1, -1, -1):
                s, a, r = episode[t]
                G = self.gamma * G + r

                # Weighted IS incremental update for Q(s,a)
                self.C[s, a] += W
                self.Q[s, a] += (W / self.C[s, a]) * (G - self.Q[s, a])

                # Policy improvement: greedy step
                self.policy[s] = int(np.argmax(self.Q[s]))

                # If behavior action != target action, IS ratio = 0 -> stop
                if a != self.policy[s]:
                    break

                # Update IS weight: pi(a|s)=1, b(a|s)=b_prob  => W *= 1/b_prob
                W *= 1.0 / self.b_prob

            if ep > 0 and ep % 10_000 == 0:
                logger.info("[P4-MC] episode %6d/%d", ep, self.n_episodes)

        elapsed = time.perf_counter() - t0

        # Extract V from Q
        self.V = np.max(self.Q, axis=1).copy()
        self.V[self.env.goal_idx]      = 0.0
        self.policy[self.env.goal_idx] = 0

        logger.info("[P4-MC] Done: %d episodes  %.3fs", self.n_episodes, elapsed)
        return self.n_episodes, elapsed
