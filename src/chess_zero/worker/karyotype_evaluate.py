"""
Evaluation worker for the karyotype correction agent.

Compares a newly-trained KaryotypeModel (next-generation) against the current
best model by running them both on a held-out set of corrupted karyotype
episodes.  Accepts the new model as "best" only if its average corrected
accuracy exceeds the current best by at least ``config.eval.accuracy_threshold``.

Mirrors the design of worker/evaluate.py.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from logging import getLogger
from multiprocessing import Manager
from time import sleep

import numpy as np

from chess_zero.agent.model_karyotype import KaryotypeModel
from chess_zero.agent.player_karyotype import KaryotypePlayer
from chess_zero.config import Config
from chess_zero.env.karyotype_env import (
    KaryotypeEnv,
    N_CHROMOSOMES,
    build_ground_truth_assignments,
    corrupt_assignments,
)

logger = getLogger(__name__)


def start(config: Config):
    return KaryotypeEvaluateWorker(config).start()


class KaryotypeEvaluateWorker:
    """
    Evaluates next-generation KaryotypeModels and promotes the best one.

    Attributes
    ----------
    config : Config
    play_config : PlayConfig
        Taken from config.eval.play_config.
    current_model : KaryotypeModel
        Currently best model.
    cur_pipes : Manager.list
        Shared pipe pool for the current best model.
    """

    def __init__(self, config: Config):
        self.config = config
        self.play_config = config.eval.play_config
        self.current_model = self._load_best_model()
        m = Manager()
        self.cur_pipes = m.list([
            self.current_model.get_pipes(self.play_config.search_threads)
            for _ in range(self.play_config.max_processes)
        ])

    def start(self):
        """Continuously evaluate next-generation models."""
        while True:
            ng_model, model_dir = self._load_next_gen_model()
            if ng_model is None:
                sleep(60)
                continue

            logger.info(f"Evaluating model from {model_dir}")
            ng_is_better = self._evaluate(ng_model)
            if ng_is_better:
                logger.info(f"New model accepted as best: {model_dir}")
                self._save_as_best(ng_model)
                self.current_model = ng_model
            else:
                logger.info(f"New model rejected.")
            self._archive_model(model_dir)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self, ng_model: KaryotypeModel) -> bool:
        """
        Run ``config.eval.game_num`` correction episodes with both models and
        compare mean final accuracy.

        Returns
        -------
        bool
            True iff the next-gen model's mean accuracy exceeds the current
            best's mean accuracy by at least ``accuracy_threshold``.
        """
        ec = self.config.eval
        m = Manager()
        ng_pipes = m.list([
            ng_model.get_pipes(self.play_config.search_threads)
            for _ in range(self.play_config.max_processes)
        ])

        current_scores = []
        ng_scores = []

        futures = []
        with ProcessPoolExecutor(
                max_workers=self.play_config.max_processes) as executor:
            for _ in range(ec.game_num):
                futures.append(executor.submit(
                    _eval_episode,
                    self.config,
                    cur_pipes=self.cur_pipes,
                    ng_pipes=ng_pipes,
                ))

            for fut in as_completed(futures):
                cur_acc, ng_acc = fut.result()
                current_scores.append(cur_acc)
                ng_scores.append(ng_acc)
                logger.debug(
                    f"  cur={cur_acc:.3f}  ng={ng_acc:.3f}  "
                    f"running mean: cur={np.mean(current_scores):.3f} "
                    f"ng={np.mean(ng_scores):.3f}"
                )

        mean_cur = float(np.mean(current_scores))
        mean_ng = float(np.mean(ng_scores))
        logger.info(
            f"Evaluation result: current={mean_cur:.4f}  "
            f"next_gen={mean_ng:.4f}  "
            f"improvement={mean_ng - mean_cur:+.4f}  "
            f"threshold={ec.accuracy_threshold}"
        )
        return (mean_ng - mean_cur) >= ec.accuracy_threshold

    # ── Model helpers ─────────────────────────────────────────────────────────

    def _load_best_model(self) -> KaryotypeModel:
        model = KaryotypeModel(self.config)
        rc = self.config.resource
        if not model.load(rc.karyotype_model_config_path,
                          rc.karyotype_model_weight_path):
            raise RuntimeError(
                "No best karyotype model found. "
                "Run karyotype_self_play first to create one.")
        return model

    def _load_next_gen_model(self):
        rc = self.config.resource
        dirs = self._get_next_gen_dirs(rc)
        if not dirs:
            logger.info("No next-gen model available; waiting …")
            return None, None
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        model = KaryotypeModel(self.config)
        config_path = os.path.join(
            model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(
            model_dir, rc.next_generation_model_weight_filename)
        model.load(config_path, weight_path)
        return model, model_dir

    def _save_as_best(self, model: KaryotypeModel):
        rc = self.config.resource
        model.save(rc.karyotype_model_config_path,
                   rc.karyotype_model_weight_path)

    def _archive_model(self, model_dir: str):
        """Move the evaluated next-gen model out of the queue directory."""
        rc = self.config.resource
        archive_dir = os.path.join(rc.karyotype_next_gen_dir, 'evaluated')
        os.makedirs(archive_dir, exist_ok=True)
        dest = os.path.join(archive_dir, os.path.basename(model_dir))
        try:
            os.rename(model_dir, dest)
        except OSError as exc:
            logger.warning(f"Could not archive model: {exc}")

    @staticmethod
    def _get_next_gen_dirs(rc) -> list:
        pattern = os.path.join(rc.karyotype_next_gen_dir,
                               rc.next_generation_model_dirname_tmpl % '*')
        return sorted(glob(pattern))


# ── Per-episode function (runs in subprocess) ─────────────────────────────────

def _eval_episode(config: Config, cur_pipes, ng_pipes) -> tuple:
    """
    Run one evaluation episode with both the current and next-gen model and
    return their final classification accuracies.

    Both models start from the *same* corrupted karyotype so the comparison
    is fair.

    Returns
    -------
    (float, float)
        (current_model_accuracy, next_gen_model_accuracy)
    """
    rng = np.random.default_rng()
    mc = config.model

    # Build a shared corrupted karyotype
    n_sex_x = 2 if rng.random() > 0.5 else 1
    ground_truth = build_ground_truth_assignments(n_sex_chromosomes_x=n_sex_x)
    n_errors = rng.integers(1, config.play_data.max_errors + 1)
    init_assignments = corrupt_assignments(ground_truth, int(n_errors), rng)
    # Use zero embeddings (evaluation only needs fair comparison, not realism)
    embeddings = np.zeros((N_CHROMOSOMES, mc.embedding_dim), dtype=np.float32)

    def _run(pipes) -> float:
        env = KaryotypeEnv(
            embeddings=embeddings,
            assignments=init_assignments.copy(),
            ground_truth=ground_truth,
            max_steps=config.eval.play_config.max_steps,
        )
        player = KaryotypePlayer(config, pipes=pipes,
                                 play_config=config.eval.play_config)
        while not env.done:
            action = player.action(env)
            env.step(action)
        return env.accuracy()

    cur_p = cur_pipes.pop()
    ng_p = ng_pipes.pop()
    try:
        cur_acc = _run(cur_p)
        ng_acc = _run(ng_p)
    finally:
        cur_pipes.append(cur_p)
        ng_pipes.append(ng_p)

    return cur_acc, ng_acc
