"""
Self-play worker for the karyotype correction agent.

Pipeline
--------
Image → Mask2Former (segmentation + classification) → RL correction (MCTS)

Mask2Former is the sole perception model: it segments each chromosome instance
and produces the initial class predictions.  The RL correction agent
(KaryotypeModel, an AlphaZero-style MCTS policy+value network) then iteratively
corrects those predictions to maximise classification accuracy.

Workflow
--------
1. Load the current best RL correction model (KaryotypeModel).
2. For each episode:
   a. Run Mask2FormerExtractor on a karyotype image to obtain per-chromosome
      visual embeddings and the initial class assignments (the starting state).
   b. Let the MCTS player (KaryotypePlayer) iteratively correct the Mask2Former
      predictions using the RL correction model.
   c. Compute the episode reward (accuracy improvement) and record
      (state_tensor, policy, reward) triples.
3. Flush the buffer to disk as JSON, to be consumed by karyotype_optimize.py.

Data sources
------------
* ``config.resource.karyotype_data_dir`` — root directory that contains COCO-format
  karyotype images.  Each image is paired with a ground-truth JSON annotation
  (compatible with the annotation format used for the Mask2Former training set).
* If no real data is available the worker falls back to *synthetic* mode: it
  generates canonical diploid karyotypes (build_ground_truth_assignments),
  uses zero embeddings, and corrupts/repairs them.  This lets the agent
  bootstrap before real Mask2Former data is available.
"""

import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from threading import Thread
from time import time

import numpy as np

from chess_zero.agent.model_karyotype import KaryotypeModel
from chess_zero.agent.player_karyotype import KaryotypePlayer
from chess_zero.config import Config
from chess_zero.env.karyotype_env import (
    KaryotypeEnv,
    N_CHROMOSOMES,
    N_CLASSES,
    build_ground_truth_assignments,
    corrupt_assignments,
)

logger = getLogger(__name__)


def start(config: Config):
    return KaryotypeSelfPlayWorker(config).start()


# ── Worker ────────────────────────────────────────────────────────────────────

class KaryotypeSelfPlayWorker:
    """
    Runs self-play episodes and writes training data to disk.

    Attributes
    ----------
    config : Config
    current_model : KaryotypeModel
    cur_pipes : Manager.list
        Shared list of pipe-pool lists (one per parallel process).
    buffer : list
        Accumulated (state, policy, reward) triples pending disk flush.
    """

    def __init__(self, config: Config):
        self.config = config
        self.current_model = self._load_model()
        m = Manager()
        self.cur_pipes = m.list([
            self.current_model.get_pipes(config.play.search_threads)
            for _ in range(config.play.max_processes)
        ])
        self.buffer = []

    def start(self):
        """Run self-play endlessly, flushing data to disk periodically."""
        self.buffer = []
        futures = deque()
        pc = self.config.play
        pd = self.config.play_data

        with ProcessPoolExecutor(max_workers=pc.max_processes) as executor:
            # Seed the pool
            for _ in range(pc.max_processes * 2):
                futures.append(
                    executor.submit(_self_play_episode, self.config,
                                    cur=self.cur_pipes))

            episode_idx = 0
            while True:
                episode_idx += 1
                t0 = time()
                data, info = futures.popleft().result()
                elapsed = time() - t0

                logger.info(
                    f"episode {episode_idx:4} | "
                    f"steps={info['steps']:3} | "
                    f"accuracy={info['accuracy']:.3f} | "
                    f"reward={info['reward']:+.3f} | "
                    f"t={elapsed:.1f}s | "
                    f"errors_in={info['n_errors']}"
                )

                self.buffer.extend(data)
                if episode_idx % pd.nb_game_in_file == 0:
                    self._flush_buffer()
                    self._reload_model_if_changed()

                futures.append(
                    executor.submit(_self_play_episode, self.config,
                                    cur=self.cur_pipes))

        if self.buffer:
            self._flush_buffer()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_model(self) -> KaryotypeModel:
        model = KaryotypeModel(self.config)
        rc = self.config.resource
        if not model.load(rc.karyotype_model_config_path,
                          rc.karyotype_model_weight_path):
            model.build()
            model.save(rc.karyotype_model_config_path,
                       rc.karyotype_model_weight_path)
        return model

    def _flush_buffer(self):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.karyotype_play_data_dir,
                            rc.karyotype_play_data_filename_tmpl % game_id)
        logger.info(f"Saving self-play data to {path}")
        thread = Thread(target=_write_data, args=(path, self.buffer))
        thread.start()
        self.buffer = []

    def _reload_model_if_changed(self):
        rc = self.config.resource
        digest = self.current_model.fetch_digest(rc.karyotype_model_weight_path)
        if digest != self.current_model.digest:
            logger.info("Best model changed – reloading")
            self.current_model = self._load_model()


# ── Per-episode function (runs in subprocess) ─────────────────────────────────

def _self_play_episode(config: Config, cur) -> tuple:
    """
    Play one correction episode and return training data.

    Parameters
    ----------
    config : Config
    cur : Manager.list
        Shared pool; one pipe-list is borrowed and returned.

    Returns
    -------
    (list, dict)
        - list of [state_tensor, policy, reward] triples
        - info dict with episode statistics
    """
    pipes = cur.pop()
    rng = np.random.default_rng()

    # ── 1. Obtain ground-truth and Mask2Former predictions ────────────────────
    probs, m2f_assignments, ground_truth = _load_sample(config, rng)

    # ── 2. Use Mask2Former's predicted assignments when available; otherwise
    #        introduce controlled random corruptions and build synthetic probs.
    if m2f_assignments is not None:
        init_assignments = m2f_assignments
        n_errors = int(np.sum(m2f_assignments != ground_truth))
    else:
        n_errors = rng.integers(1, config.play_data.max_errors + 1)
        init_assignments = corrupt_assignments(ground_truth, int(n_errors), rng)
        probs = _make_synthetic_probs(init_assignments, rng)

    # ── 3. Build environment ──────────────────────────────────────────────────
    env = KaryotypeEnv(
        probs=probs,
        assignments=init_assignments,
        ground_truth=ground_truth,
        max_steps=config.play.max_steps,
    )

    # ── 4. MCTS player corrects the karyotype ─────────────────────────────────
    player = KaryotypePlayer(config, pipes=pipes)

    while not env.done:
        action_idx = player.action(env)
        env.step(action_idx)

    # ── 5. Compute reward and store in move list ──────────────────────────────
    reward = env.reward()
    player.finish_episode(reward)

    info = {
        "steps": env.step_count,
        "accuracy": env.accuracy(),
        "reward": reward,
        "n_errors": int(n_errors),
    }

    cur.append(pipes)
    return player.moves, info


def _load_sample(config: Config, rng: np.random.Generator):
    """
    Load (probs, m2f_assignments, ground_truth) for one karyotype sample.

    Strategy:
    1. If a Mask2Former extractor is available and the data directory
       contains images, extract real class-probability distributions and use
       Mask2Former's predicted class assignments as the initial state.
    2. Otherwise fall back to synthetic data: return ``(None, None, ground_truth)``
       so the caller can corrupt the ground truth and build synthetic probs.
    """
    rc = config.resource
    data_dir = getattr(rc, 'karyotype_data_dir', None)
    annotation_file = getattr(rc, 'karyotype_annotation_file', None)

    if data_dir and annotation_file and os.path.isfile(annotation_file):
        try:
            return _load_real_sample(config, rng, data_dir, annotation_file)
        except Exception as exc:
            logger.warning(f"Real sample load failed ({exc}), using synthetic.")

    # Synthetic fallback — probs will be generated from corrupted assignments
    n_sex_x = 2 if rng.random() > 0.5 else 1
    ground_truth = build_ground_truth_assignments(n_sex_chromosomes_x=n_sex_x)
    return None, None, ground_truth


def _load_real_sample(config: Config, rng: np.random.Generator,
                      data_dir: str, annotation_file: str):
    """
    Load one real karyotype sample using Mask2Former predictions.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        probs of shape (N_CHROMOSOMES, N_CLASSES) — per-chromosome class
            probability distributions from Mask2Former (frozen initial state),
        mask2former_assignments of shape (N_CHROMOSOMES,) — predicted class
            labels (1–24), argmax of probs, used as the initial RL assignment,
        ground_truth labels of shape (N_CHROMOSOMES,).
    """
    from chess_zero.lib.mask2former_extractor import Mask2FormerExtractor

    rc = config.resource
    extractor = Mask2FormerExtractor(
        config_file=rc.mask2former_config_file,
        checkpoint_file=rc.mask2former_checkpoint_file,
        device=getattr(rc, 'device', 'cpu'),
        cnsn_model_dir=getattr(rc, 'cnsn_model_dir', None),
    )

    with open(annotation_file, 'rt') as f:
        coco_ann = json.load(f)

    images = coco_ann['images']
    img_meta = images[int(rng.integers(len(images)))]
    image_path = os.path.join(data_dir, img_meta['file_name'])

    result = extractor.extract(image_path)

    # Build ground-truth from COCO annotations for this image
    img_id = img_meta['id']
    cat_id_to_class = {
        cat['id']: int(cat['name'].split('_')[1])
        for cat in coco_ann['categories']
    }
    anns = [a for a in coco_ann['annotations'] if a['image_id'] == img_id]

    ground_truth = np.ones(N_CHROMOSOMES, dtype=np.int32)
    for i, ann in enumerate(anns[:N_CHROMOSOMES]):
        ground_truth[i] = cat_id_to_class.get(ann['category_id'], 1)

    # Use Mask2Former's class probabilities and argmax assignments
    return result.probs, result.assignments, ground_truth


def _make_synthetic_probs(assignments: np.ndarray,
                           rng: np.random.Generator,
                           confidence: float = 0.7) -> np.ndarray:
    """
    Build synthetic per-chromosome class probability vectors for the
    synthetic-data fallback.

    Each chromosome is assigned a probability of ``confidence`` on its
    current class, with the remaining mass distributed uniformly across
    the other N_CLASSES-1 classes.  This mimics a reasonably confident
    but not perfect initial classifier output.

    Parameters
    ----------
    assignments : np.ndarray, shape (N_CHROMOSOMES,)
        Current class labels (1-indexed).
    rng : np.random.Generator
        Random number generator (not used currently, reserved for future
        Dirichlet noise injection).
    confidence : float
        Probability mass placed on the assigned class (default 0.7).

    Returns
    -------
    np.ndarray, shape (N_CHROMOSOMES, N_CLASSES)
    """
    residual = (1.0 - confidence) / (N_CLASSES - 1)
    probs = np.full((N_CHROMOSOMES, N_CLASSES), residual, dtype=np.float32)
    for i, cls in enumerate(assignments):
        probs[i, int(np.clip(cls - 1, 0, N_CLASSES - 1))] = confidence
    return probs


def _write_data(path: str, data: list):
    """Write training data to a JSON file (runs in a background thread)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'wt') as f:
            json.dump(
                [[s.tolist(), p, r] for s, p, r in data],
                f,
            )
    except Exception as exc:
        logger.error(f"Failed to write data to {path}: {exc}")
