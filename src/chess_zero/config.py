"""
Everything related to configuration of running this application
"""

import os
import numpy as np


class PlayWithHumanConfig:
    """
    Config for allowing human to play against an agent using uci

    """
    def __init__(self):
        self.simulation_num_per_move = 1200
        self.threads_multiplier = 2
        self.c_puct = 1 # lower  = prefer mean action value
        self.noise_eps = 0
        self.tau_decay_rate = 0  # start deterministic mode
        self.resign_threshold = None

    def update_play_config(self, pc):
        """
        :param PlayConfig pc:
        :return:
        """
        pc.simulation_num_per_move = self.simulation_num_per_move
        pc.search_threads *= self.threads_multiplier
        pc.c_puct = self.c_puct
        pc.noise_eps = self.noise_eps
        pc.tau_decay_rate = self.tau_decay_rate
        pc.resign_threshold = self.resign_threshold
        pc.max_game_length = 999999


class Options:
    new = False


class ResourceConfig:
    """
    Config describing all of the directories and resources needed during running this project
    """
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())

        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.model_best_config_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")

        self.model_best_distributed_ftp_server = "alpha-chess-zero.mygamesonline.org"
        self.model_best_distributed_ftp_user = "2537576_chess"
        self.model_best_distributed_ftp_password = "alpha-chess-zero-2"
        self.model_best_distributed_ftp_remote_path = "/alpha-chess-zero.mygamesonline.org/"

        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"

        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"

        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")

        # ── Karyotype correction agent resources ──────────────────────────────
        # Root directory for karyotype data (COCO-format images + annotations).
        # Override via the KARYOTYPE_DATA_DIR environment variable.
        self.karyotype_data_dir = os.environ.get(
            "KARYOTYPE_DATA_DIR",
            os.path.join(self.data_dir, "karyotype"))

        # Path to the COCO annotation JSON for ground-truth labels.
        self.karyotype_annotation_file = os.environ.get(
            "KARYOTYPE_ANN_FILE",
            os.path.join(self.karyotype_data_dir, "annotations", "val_500.json"))

        # ── Mask2Former — the sole perception model ────────────────────────────
        # Mask2Former handles BOTH instance segmentation and initial chromosome
        # classification.  Its outputs (per-chromosome visual embeddings + initial
        # class predictions) are the direct input to the RL correction agent.
        # No separate chromosome classification model is needed.
        #
        # Mask2Former mmdetection config file
        # (default: the config committed in mask2former/).
        self.mask2former_config_file = os.environ.get(
            "MASK2FORMER_CONFIG",
            os.path.join(self.project_dir,
                         "mask2former", "cnsn_resnet50_mcls_6k.py"))

        # Mask2Former checkpoint (.pth) file.
        self.mask2former_checkpoint_file = os.environ.get(
            "MASK2FORMER_CHECKPOINT", "")

        # Directory containing the cnsn_models package required by the backbone.
        self.cnsn_model_dir = os.environ.get(
            "CNSN_MODEL_DIR",
            os.path.join(self.project_dir, "mask2former"))

        # ── RL correction model (policy + value network) ───────────────────────
        # This is the AlphaZero-style MCTS correction agent — NOT a chromosome
        # classifier.  It takes as input the state produced by Mask2Former
        # (visual embeddings + current class-assignment one-hot vectors) and
        # outputs a correction policy and value estimate used by MCTS.
        # Saved as the "best accepted" version after each evaluation round.
        self.karyotype_model_dir = os.path.join(self.karyotype_data_dir, "model")
        self.karyotype_model_config_path = os.path.join(
            self.karyotype_model_dir, "karyotype_model_config.json")
        self.karyotype_model_weight_path = os.path.join(
            self.karyotype_model_dir, "karyotype_model_weight.h5")

        # Next-generation candidates produced by karyotype_optimize.py.
        self.karyotype_next_gen_dir = os.path.join(
            self.karyotype_model_dir, "next_generation")

        # Self-play data written by karyotype_self_play.py.
        self.karyotype_play_data_dir = os.path.join(
            self.karyotype_data_dir, "play_data")
        self.karyotype_play_data_filename_tmpl = "karyotype_play_%s.json"

        # Torch device for Mask2Former inference.
        # Prefer GPU when available; fall back to CPU automatically.
        # Override by setting the DEVICE environment variable (e.g. DEVICE=cpu).
        if "DEVICE" in os.environ:
            self.device = os.environ["DEVICE"]
        else:
            try:
                import torch
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir,
                self.karyotype_data_dir, self.karyotype_model_dir,
                self.karyotype_next_gen_dir, self.karyotype_play_data_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


def flipped_uci_labels():
    """
    Seems to somehow transform the labels used for describing the universal chess interface format, putting
    them into a returned list.
    :return:
    """
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in create_uci_labels()]


def create_uci_labels():
    """
    Creates the labels for the universal chess interface into an array and returns them
    :return:
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                           [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + '2' + l + '1' + p)
            labels_array.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + '2' + l_l + '1' + p)
                labels_array.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + '2' + l_r + '1' + p)
                labels_array.append(l + '7' + l_r + '8' + p)
    return labels_array


class Config:
    """
    Config describing how to run the application

    Attributes (best guess so far):
        :ivar list(str) labels: labels to use for representing the game using UCI
        :ivar int n_lables: number of labels
        :ivar list(str) flipped_labels: some transformation of the labels
        :ivar int unflipped_index: idk
        :ivar Options opts: options to use to configure this config
        :ivar ResourceConfig resources: resources used by this config.
        :ivar ModelConfig mode: config for the model to use
        :ivar PlayConfig play: configuration for the playing of the game
        :ivar PlayDataConfig play_date: configuration for the saved data from playing
        :ivar TrainerConfig trainer: config for how training should go
        :ivar EvaluateConfig eval: config for how evaluation should be done
    """
    labels = create_uci_labels()
    n_labels = int(len(labels))
    flipped_labels = flipped_uci_labels()
    unflipped_index = None

    def __init__(self, config_type="mini"):
        """

        :param str config_type: one of "mini", "normal", or "distributed", representing the set of
            configs to use for all of the config attributes. Mini is a small version, normal is the
            larger version, and distributed is a version which runs across multiple GPUs it seems
        """
        self.opts = Options()
        self.resource = ResourceConfig()

        if config_type == "mini":
            import chess_zero.configs.mini as c
        elif config_type == "normal":
            import chess_zero.configs.normal as c
        elif config_type == "distributed":
            import chess_zero.configs.distributed as c
        elif config_type == "karyotype":
            import chess_zero.configs.karyotype as c
        else:
            raise RuntimeError(f"unknown config_type: {config_type}")
        self.model = c.ModelConfig()
        self.play = c.PlayConfig()
        self.play_data = c.PlayDataConfig()
        self.trainer = c.TrainerConfig()
        self.eval = c.EvaluateConfig()
        if config_type != "karyotype":
            self.labels = Config.labels
            self.n_labels = Config.n_labels
            self.flipped_labels = Config.flipped_labels

    @staticmethod
    def flip_policy(pol):
        """

        :param pol policy to flip:
        :return: the policy, flipped (for switching between black and white it seems)
        """
        return np.asarray([pol[ind] for ind in Config.unflipped_index])


Config.unflipped_index = [Config.labels.index(x) for x in Config.flipped_labels]


# print(Config.labels)
# print(Config.flipped_labels)


def _project_dir():
    d = os.path.dirname
    return d(d(d(os.path.abspath(__file__))))


def _data_dir():
    return os.path.join(_project_dir(), "data")