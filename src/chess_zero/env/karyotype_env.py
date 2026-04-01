"""
Karyotype correction environment for the AlphaZero-style RL agent.

Replaces chess_env.py: the "board" is a karyotype state consisting of
  - per-chromosome class probability distributions (from Mask2Former, frozen)
  - current class assignment for each chromosome (1–24)

The RL state is a flat vector of dimension STATE_DIM = 1198:

  (1) Predicted probabilities (frozen, from Mask2Former):
        46 chromosomes × 24 classes = 1104 floats
  (2) Current hard assignments, normalised:
        46 floats (each assignment / N_CLASSES, range [1/24, 1])
  (3) Current class counts, normalised:
        24 floats (count_per_class / N_CHROMOSOMES)
  (4) Binary constraint violation indicators:
        24 floats (1 if class count violates diploid rules, else 0)

  Total: 1104 + 46 + 24 + 24 = 1198

The predicted probabilities are the direct output of Mask2Former and are
*never updated* during an episode — they represent the classifier's initial
confidence.  Components (2)–(4) are updated after each correction action.

Actions are (chromosome_index, target_class) reassignment operations, plus
a special STOP action that ends the episode.

Medical prior constraints (diploid rules) are enforced when generating the set
of legal actions, mirroring how python-chess exposes legal_moves.
"""

import copy
from logging import getLogger

import numpy as np

logger = getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
N_CHROMOSOMES = 46          # human karyotype
N_CLASSES = 24              # chromosome_01 … chromosome_22, X (23), Y (24)
N_AUTOSOME_CLASSES = 22     # classes 1–22 are autosomes
X_CLASS = 23                # class index for X chromosome
Y_CLASS = 24                # class index for Y chromosome

# Action encoding:
#   index = chr_i * N_CLASSES + (target_class - 1)   for reassign
#   index = N_CHROMOSOMES * N_CLASSES                 for STOP
N_ACTIONS = N_CHROMOSOMES * N_CLASSES + 1
STOP_ACTION = N_ACTIONS - 1  # == 1104

# Flat RL state dimension:
#   probs (1104) + hard assignments (46) + class counts (24) + violations (24)
STATE_DIM = N_CHROMOSOMES * N_CLASSES + N_CHROMOSOMES + N_CLASSES + N_CLASSES  # 1198


# ── Helper functions ─────────────────────────────────────────────────────────

def action_to_index(chr_i: int, target_class: int) -> int:
    """Encode a (chromosome, class) pair into a flat action index."""
    return chr_i * N_CLASSES + (target_class - 1)


def index_to_action(idx: int):
    """Decode a flat action index.

    Returns ('stop', None, None) for the STOP action, or
    ('reassign', chr_i, target_class) otherwise.
    """
    if idx == STOP_ACTION:
        return 'stop', None, None
    chr_i = idx // N_CLASSES
    target_class = (idx % N_CLASSES) + 1
    return 'reassign', chr_i, target_class


# ── Main environment class ────────────────────────────────────────────────────

class KaryotypeEnv:
    """
    Models the chromosome karyotype correction problem as an RL environment.

    Attributes:
        probs (np.ndarray): shape (N_CHROMOSOMES, N_CLASSES) – per-class
            probability distributions output by Mask2Former for each
            chromosome.  These are *frozen* for the entire episode.
        assignments (np.ndarray): shape (N_CHROMOSOMES,) – current class label
            (1–24) assigned to each chromosome.  Updated by each action.
        ground_truth (np.ndarray | None): shape (N_CHROMOSOMES,) – correct class
            labels.  None during inference.
        step_count (int): number of correction actions taken so far.
        stopped (bool): True when STOP action was chosen.
        max_steps (int): episode length limit.
    """

    def __init__(self, probs: np.ndarray, assignments: np.ndarray,
                 ground_truth=None, max_steps: int = 20):
        """
        Parameters
        ----------
        probs : np.ndarray, shape (N_CHROMOSOMES, N_CLASSES)
            Per-chromosome class probability distributions from Mask2Former
            (or synthetic equivalents).  These are frozen for the episode.
        assignments : np.ndarray, shape (N_CHROMOSOMES,)
            Initial class labels (integers in 1–24) produced by the initial
            classifier (Mask2Former).
        ground_truth : np.ndarray or None
            Correct class labels; required during training, optional at inference.
        max_steps : int
            Maximum correction steps before the episode is forced to end.
        """
        self.probs = np.asarray(probs, dtype=np.float32)
        self.probs.flags.writeable = False  # frozen for entire episode
        self.assignments = np.asarray(assignments, dtype=np.int32)
        self.ground_truth = (np.asarray(ground_truth, dtype=np.int32)
                             if ground_truth is not None else None)
        self.max_steps = max_steps
        self.step_count = 0
        self.stopped = False

        assert self.probs.shape == (N_CHROMOSOMES, N_CLASSES), \
            f"probs must have shape ({N_CHROMOSOMES}, {N_CLASSES}), got {self.probs.shape}"
        assert self.assignments.shape == (N_CHROMOSOMES,)
        assert np.all((self.assignments >= 1) & (self.assignments <= N_CLASSES)), \
            "All assignments must be in 1..N_CLASSES"

    # ── Core interface ────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        """True when the episode has ended (STOP chosen or step limit reached)."""
        return self.stopped or self.step_count >= self.max_steps

    def step(self, action_idx: int):
        """Apply one correction action.

        Parameters
        ----------
        action_idx : int
            Flat action index (0 … N_ACTIONS-1).
        """
        if self.done:
            raise RuntimeError("step() called on a finished episode")
        kind, chr_i, target_class = index_to_action(action_idx)
        if kind == 'stop':
            self.stopped = True
        else:
            self.assignments[chr_i] = target_class
            self.step_count += 1

    def reward(self) -> float:
        """
        Compute the episode reward as a value in [-1, +1].

        Maps accuracy (fraction of correctly assigned chromosomes) linearly:
            accuracy 1.0  →  reward +1.0
            accuracy 0.0  →  reward -1.0
        """
        if self.ground_truth is None:
            raise RuntimeError("Cannot compute reward without ground_truth")
        correct = np.sum(self.assignments == self.ground_truth)
        accuracy = correct / N_CHROMOSOMES
        return 2.0 * accuracy - 1.0  # in [-1, +1]

    def legal_action_indices(self, play_config=None) -> list:
        """
        Return a list of legal flat action indices based on medical priors.

        Rules enforced:
        1. Autosome classes (1–22): at most ``max_autosome_count`` (default 2)
           chromosomes may be assigned to each class.
        2. X chromosome class (23): at most ``max_x_count`` (default 2).
        3. Y chromosome class (24): at most ``max_y_count`` (default 1).
        4. Reassigning a chromosome to its *current* class is a no-op → illegal.
        5. STOP is always legal.

        Parameters
        ----------
        play_config : PlayConfig | None
            If provided, uses its ``max_autosome_count``, ``max_x_count`` and
            ``max_y_count`` limits.  Falls back to biological defaults (2, 2, 1).
        """
        max_auto = getattr(play_config, 'max_autosome_count', 2)
        max_x = getattr(play_config, 'max_x_count', 2)
        max_y = getattr(play_config, 'max_y_count', 1)

        # Count current occupancy per class
        class_counts = np.zeros(N_CLASSES + 1, dtype=np.int32)  # 1-indexed
        for cls in self.assignments:
            class_counts[cls] += 1

        legal = []
        for i in range(N_CHROMOSOMES):
            current_cls = int(self.assignments[i])
            for c in range(1, N_CLASSES + 1):
                if c == current_cls:
                    continue  # no-op reassignment
                # Compute occupancy *after* removing chr i from its current class
                # and placing it in class c
                new_count_c = class_counts[c] + 1
                if 1 <= c <= N_AUTOSOME_CLASSES:
                    if new_count_c > max_auto:
                        continue  # would overfill an autosome class
                elif c == X_CLASS:
                    if new_count_c > max_x:
                        continue
                elif c == Y_CLASS:
                    if new_count_c > max_y:
                        continue
                legal.append(action_to_index(i, c))

        legal.append(STOP_ACTION)
        return legal

    def encode_state(self) -> np.ndarray:
        """
        Encode the current karyotype state into a flat vector for the neural
        network.

        Returns
        -------
        np.ndarray, shape (STATE_DIM,) = (1198,)
            Concatenation of four components:

            (1) Predicted probabilities (frozen, from Mask2Former):
                  self.probs.flatten() — 46 × 24 = 1104 floats.
                  Never updated during the episode.

            (2) Normalised hard assignments:
                  self.assignments / N_CLASSES — 46 floats in [1/24, 1].
                  Updated after each correction action.

            (3) Normalised class counts:
                  count[c] / N_CHROMOSOMES for c in 1..N_CLASSES — 24 floats.
                  count[c] = number of chromosomes currently assigned to class c.

            (4) Binary constraint violation indicators:
                  1.0 if the current count for class c violates diploid rules,
                  else 0.0 — 24 floats.
                  Rules (diploid defaults):
                    · Autosomes (1–22): violation iff count ≠ 2
                    · X (class 23):    violation iff count > 2
                    · Y (class 24):    violation iff count > 1
        """
        # (1) Frozen predicted probabilities → shape (1104,)
        probs_flat = self.probs.flatten()

        # (2) Normalised current assignments → shape (46,)
        assignments_norm = self.assignments.astype(np.float32) / N_CLASSES

        # Compute per-class chromosome counts (used for components 3 and 4)
        counts = np.zeros(N_CLASSES, dtype=np.float32)
        for cls in self.assignments:
            counts[cls - 1] += 1.0  # cls is 1-indexed

        # (3) Normalised class counts → shape (24,)
        counts_norm = counts / N_CHROMOSOMES

        # (4) Binary constraint violation indicators → shape (24,)
        violations = np.zeros(N_CLASSES, dtype=np.float32)
        for c in range(1, N_CLASSES + 1):
            cnt = counts[c - 1]
            if 1 <= c <= N_AUTOSOME_CLASSES:
                violations[c - 1] = 1.0 if cnt != 2.0 else 0.0
            elif c == X_CLASS:
                violations[c - 1] = 1.0 if cnt > 2.0 else 0.0
            else:  # Y_CLASS
                violations[c - 1] = 1.0 if cnt > 1.0 else 0.0

        state = np.concatenate([probs_flat, assignments_norm,
                                 counts_norm, violations])
        assert state.shape == (STATE_DIM,), \
            f"encode_state: expected shape ({STATE_DIM},), got {state.shape}"
        return state

    def copy(self) -> 'KaryotypeEnv':
        """Return a deep copy of this environment for MCTS tree search."""
        env = KaryotypeEnv.__new__(KaryotypeEnv)
        env.probs = self.probs        # frozen (write-protected); safe to share reference
        env.assignments = self.assignments.copy()
        env.ground_truth = self.ground_truth  # read-only; share reference
        env.max_steps = self.max_steps
        env.step_count = self.step_count
        env.stopped = self.stopped
        return env

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def observation(self) -> str:
        """
        A string key uniquely identifying the current karyotype state,
        analogous to FEN in chess.  Used as the MCTS tree node key.
        """
        return ','.join(map(str, self.assignments.tolist()))

    def accuracy(self) -> float:
        """Fraction of chromosomes correctly classified (requires ground_truth)."""
        if self.ground_truth is None:
            return float('nan')
        return float(np.mean(self.assignments == self.ground_truth))

    def render(self):
        """Print a simple textual representation of the current state."""
        print(f"\nStep {self.step_count}/{self.max_steps}")
        if self.ground_truth is not None:
            correct = np.sum(self.assignments == self.ground_truth)
            print(f"Accuracy: {correct}/{N_CHROMOSOMES} = {correct/N_CHROMOSOMES:.2%}")
        class_map = {}
        for i, cls in enumerate(self.assignments):
            class_map.setdefault(cls, []).append(i)
        for cls in sorted(class_map):
            print(f"  Class {cls:2d}: chromosomes {class_map[cls]}")
        print()


# ── Data corruption utilities (used by self-play worker) ─────────────────────

def corrupt_assignments(ground_truth: np.ndarray, n_swaps: int,
                        rng=None) -> np.ndarray:
    """
    Introduce controlled errors into a correct karyotype assignment by randomly
    swapping the class labels of ``n_swaps`` pairs of chromosomes.

    This mimics the kinds of mistakes a primary classifier (e.g. Mask2Former)
    might make on morphologically similar chromosomes.

    Parameters
    ----------
    ground_truth : np.ndarray, shape (N_CHROMOSOMES,)
        Correct class labels.
    n_swaps : int
        Number of (i, j) swaps to perform.
    rng : np.random.Generator | None
        Optional random number generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (N_CHROMOSOMES,)
        Corrupted assignments (copy of ground_truth with swaps applied).
    """
    if rng is None:
        rng = np.random.default_rng()
    corrupted = ground_truth.copy()
    indices = np.arange(N_CHROMOSOMES)
    for _ in range(n_swaps):
        i, j = rng.choice(indices, size=2, replace=False)
        corrupted[i], corrupted[j] = corrupted[j], corrupted[i]
    return corrupted


def build_ground_truth_assignments(n_sex_chromosomes_x: int = 2) -> np.ndarray:
    """
    Build a canonical ground-truth assignment array for a normal human karyotype.

    Parameters
    ----------
    n_sex_chromosomes_x : int
        2 for female (XX), 1 for male (XY).  Defaults to 2 (female).

    Returns
    -------
    np.ndarray, shape (N_CHROMOSOMES,)
        Class labels (1–24) for 46 chromosomes in canonical order.
    """
    gt = []
    for cls in range(1, N_AUTOSOME_CLASSES + 1):  # classes 1–22, 2 each
        gt.extend([cls, cls])
    # Sex chromosomes
    for _ in range(n_sex_chromosomes_x):
        gt.append(X_CLASS)  # 23
    if n_sex_chromosomes_x < 2:
        gt.append(Y_CLASS)  # 24 for XY male
    assert len(gt) == N_CHROMOSOMES, f"Expected {N_CHROMOSOMES} chromosomes, got {len(gt)}"
    return np.array(gt, dtype=np.int32)
