"""
MCTS-based chromosome karyotype correction player.

Mirrors the structure of player_chess.py but operates on KaryotypeEnv:

  - State:   encoded karyotype tensor (N_CHROMOSOMES, embedding_dim + N_CLASSES)
  - Actions: flat indices 0 … N_ACTIONS-1 (reassign or STOP)
  - Legal actions are filtered by biological/medical priors
  - Tree search terminates when the episode is "done" (STOP or step limit)

The search accumulates visit statistics (N, W, Q, P) for each (state, action) pair
exactly as in the AlphaGo Zero paper, then derives a policy from visit counts.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock

import numpy as np

from chess_zero.env.karyotype_env import (
    KaryotypeEnv, STOP_ACTION, N_ACTIONS,
)

logger = getLogger(__name__)


# ── Visit statistics (same design as player_chess.py) ────────────────────────

class VisitStats:
    """Stats for all actions from one game state."""
    def __init__(self):
        self.a = defaultdict(ActionStats)   # action_idx → ActionStats
        self.p = None                        # prior probabilities array (set once)
        self.sum_n = 0


class ActionStats:
    """Stats for one (state, action) pair."""
    def __init__(self):
        self.n = 0    # visit count
        self.w = 0.0  # total value
        self.q = 0.0  # mean value  (w / n)
        self.p = 0.0  # prior probability


# ── Player ────────────────────────────────────────────────────────────────────

class KaryotypePlayer:
    """
    MCTS-based player for the chromosome karyotype correction task.

    At each step it runs ``simulation_num_per_move`` MCTS simulations (spread
    across ``search_threads`` threads), derives a policy from visit counts, and
    picks the next correction action.

    Parameters
    ----------
    config : Config
        Global config.  Uses ``config.play`` (PlayConfig from karyotype.py) and
        ``config.model`` (ModelConfig) for action-space dimensions.
    pipes : list[Connection] | None
        Pool of pipe connections to the KaryotypeModelAPI prediction server.
    play_config : PlayConfig | None
        Override config.play.
    dummy : bool
        If True, skips pipe setup (for supervised / evaluation use).
    """

    def __init__(self, config, pipes=None, play_config=None, dummy=False):
        self.config = config
        self.play_config = play_config or config.play
        self.n_actions = config.model.n_actions()   # 1105
        self.moves = []   # list of (state_tensor, policy, reward) collected during episode

        self.tree = defaultdict(VisitStats)
        self.node_lock = defaultdict(Lock)

        if dummy:
            return

        self.pipe_pool = pipes

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self):
        """Clear the MCTS tree to start a new episode."""
        self.tree = defaultdict(VisitStats)

    def action(self, env: KaryotypeEnv) -> int:
        """
        Pick the next correction action via MCTS.

        Parameters
        ----------
        env : KaryotypeEnv
            Current environment state.

        Returns
        -------
        int
            Flat action index (0 … N_ACTIONS-1).
        """
        self.reset()
        self.search_moves(env)
        policy = self.calc_policy(env)
        action_idx = int(np.random.choice(
            self.n_actions,
            p=self.apply_temperature(policy, env.step_count),
        ))
        self.moves.append([env.encode_state(), policy.tolist()])
        return action_idx

    def finish_episode(self, reward: float):
        """
        Append the final reward to every recorded move in this episode.

        Parameters
        ----------
        reward : float
            Episode reward in [-1, +1].
        """
        for move in self.moves:
            move.append(reward)

    # ── MCTS core ─────────────────────────────────────────────────────────────

    def search_moves(self, env: KaryotypeEnv):
        """
        Run ``simulation_num_per_move`` MCTS simulations in parallel threads.
        """
        pc = self.play_config
        with ThreadPoolExecutor(max_workers=pc.search_threads) as executor:
            futures = [
                executor.submit(self._search, env.copy(), is_root=True)
                for _ in range(pc.simulation_num_per_move)
            ]
        _ = [f.result() for f in futures]

    def _search(self, env: KaryotypeEnv, is_root: bool = False) -> float:
        """
        One MCTS simulation. Returns the value estimate from the current
        player's point of view (higher = better for the corrector).

        Parameters
        ----------
        env : KaryotypeEnv
            A copy of the environment to simulate in.
        is_root : bool
            Whether this is the root call of a simulation.

        Returns
        -------
        float
            Value estimate in [-1, +1].
        """
        # Terminal: episode done
        if env.done:
            # Return intermediate reward: accuracy-based
            if env.ground_truth is not None:
                return env.reward()
            return 0.0

        state_key = env.observation
        pc = self.play_config

        with self.node_lock[state_key]:
            if state_key not in self.tree:
                # Leaf: expand and evaluate
                leaf_p, leaf_v = self._expand_and_evaluate(env)
                self.tree[state_key].p = leaf_p
                return leaf_v

            # SELECT
            action_idx = self._select_action(env, is_root)

            virtual_loss = pc.virtual_loss
            visit = self.tree[state_key]
            a_stats = visit.a[action_idx]

            visit.sum_n += virtual_loss
            a_stats.n += virtual_loss
            a_stats.w += -virtual_loss
            a_stats.q = a_stats.w / a_stats.n

        # Step in the simulation copy
        env.step(action_idx)
        leaf_v = self._search(env)

        # BACKUP
        with self.node_lock[state_key]:
            visit.sum_n += -virtual_loss + 1
            a_stats.n += -virtual_loss + 1
            a_stats.w += virtual_loss + leaf_v
            a_stats.q = a_stats.w / a_stats.n

        return leaf_v

    def _expand_and_evaluate(self, env: KaryotypeEnv):
        """
        Query the policy+value network for the current state.

        Returns
        -------
        (np.ndarray, float)
            Prior policy over N_ACTIONS actions and scalar value estimate.
        """
        state_tensor = env.encode_state()  # (N_CHROMOSOMES, input_dim)
        policy, value = self._predict(state_tensor)

        # Zero-out illegal actions and renormalise
        legal = set(env.legal_action_indices(self.play_config))
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for idx in legal:
            mask[idx] = 1.0
        policy = policy * mask
        total = policy.sum()
        if total > 1e-8:
            policy /= total
        else:
            # Uniform over legal actions as fallback
            policy = mask / mask.sum()

        return policy, value

    def _predict(self, state_tensor: np.ndarray):
        """
        Send state tensor down the pipe and receive (policy, value).
        """
        pipe = self.pipe_pool.pop()
        pipe.send(state_tensor)
        result = pipe.recv()
        self.pipe_pool.append(pipe)
        return result  # (policy_array, float)

    def _select_action(self, env: KaryotypeEnv, is_root: bool) -> int:
        """
        Select the action maximising Q + U (AGZ UCB formula).

        Illegal actions receive Q + U = -inf so they are never selected.

        Parameters
        ----------
        env : KaryotypeEnv
        is_root : bool
            Add Dirichlet noise at the root for exploration.

        Returns
        -------
        int
            Selected action index.
        """
        state_key = env.observation
        visit = self.tree[state_key]
        pc = self.play_config

        # Push prior probabilities to edges on first visit
        if visit.p is not None:
            legal = env.legal_action_indices(self.play_config)
            tot_p = 1e-8
            for idx in legal:
                p_val = visit.p[idx]
                visit.a[idx].p = p_val
                tot_p += p_val
            for a_s in visit.a.values():
                a_s.p /= tot_p
            visit.p = None

        sqrt_n = np.sqrt(visit.sum_n + 1)
        e = pc.noise_eps
        c_puct = pc.c_puct
        dir_alpha = pc.dirichlet_alpha

        legal_set = set(env.legal_action_indices(self.play_config))

        if is_root and legal_set:
            noise = np.random.dirichlet(
                [dir_alpha] * len(legal_set))
            noise_map = dict(zip(sorted(legal_set), noise))
        else:
            noise_map = {}

        best_score = -np.inf
        best_action = STOP_ACTION  # safe default

        for idx in legal_set:
            a_s = visit.a[idx]
            p_ = a_s.p
            if is_root and idx in noise_map:
                p_ = (1 - e) * p_ + e * noise_map[idx]
            score = a_s.q + c_puct * p_ * sqrt_n / (1 + a_s.n)
            if score > best_score:
                best_score = score
                best_action = idx

        return best_action

    # ── Policy and temperature ────────────────────────────────────────────────

    def calc_policy(self, env: KaryotypeEnv) -> np.ndarray:
        """
        Derive a policy vector from MCTS visit counts.

        Returns
        -------
        np.ndarray, shape (N_ACTIONS,)
            Normalised visit counts.
        """
        state_key = env.observation
        visit = self.tree[state_key]
        policy = np.zeros(self.n_actions, dtype=np.float32)
        for idx, a_s in visit.a.items():
            policy[idx] = a_s.n
        total = policy.sum()
        if total > 0:
            policy /= total
        else:
            policy[STOP_ACTION] = 1.0  # fallback: just STOP
        return policy

    def apply_temperature(self, policy: np.ndarray, step: int) -> np.ndarray:
        """
        Apply temperature-based randomisation to the policy.

        High temperature → more random; tau → 0 → deterministic (argmax).

        Parameters
        ----------
        policy : np.ndarray
        step : int
            Number of correction steps taken so far in this episode.

        Returns
        -------
        np.ndarray
            Probability distribution over N_ACTIONS.
        """
        tau = np.power(self.play_config.tau_decay_rate, step + 1)
        if tau < 0.1:
            tau = 0.0
        if tau == 0:
            best = int(np.argmax(policy))
            ret = np.zeros(self.n_actions, dtype=np.float32)
            ret[best] = 1.0
            return ret
        ret = np.power(policy + 1e-10, 1.0 / tau)
        ret /= ret.sum()
        return ret
