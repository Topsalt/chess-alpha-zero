"""
Configuration for the karyotype chromosome classification correction agent.

Mirrors the structure of configs/normal.py and configs/mini.py but tuned for
the chromosome karyotype correction task using AlphaZero-style MCTS + RL.
"""


class EvaluateConfig:
    def __init__(self):
        self.vram_frac = 1.0
        # Number of corrupted test cases to compare new vs. best model
        self.game_num = 50
        # Accept new model only if it beats best model by this accuracy margin
        self.accuracy_threshold = 0.02
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 100
        self.play_config.thinking_loop = 1
        self.play_config.c_puct = 1.0
        self.play_config.tau_decay_rate = 0.0  # deterministic during eval
        self.play_config.noise_eps = 0
        self.evaluate_latest_first = True


class PlayDataConfig:
    def __init__(self):
        # How many repair episodes to pack into a single data file
        self.nb_game_in_file = 50
        # Maximum number of data files to keep on disk
        self.max_file_num = 200
        # Maximum number of swaps to introduce when generating a corrupted karyotype
        self.max_errors = 5


class PlayConfig:
    def __init__(self):
        # Parallel self-repair processes
        self.max_processes = 2
        # Threads for MCTS per process (shared pipe pool)
        self.search_threads = 8
        self.vram_frac = 1.0
        # MCTS simulations per correction step
        self.simulation_num_per_move = 200
        self.thinking_loop = 1
        self.logging_thinking = False
        # UCB exploration constant
        self.c_puct = 1.5
        # Dirichlet noise for root exploration
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3
        # Temperature decay (1.0 = always stochastic, 0.0 = deterministic)
        self.tau_decay_rate = 0.99
        # Virtual loss to discourage thread collisions in tree
        self.virtual_loss = 3
        # Maximum correction steps per episode (prevents infinite loops)
        self.max_steps = 20

        # ── Medical prior constraints ─────────────────────────────────────────
        # Normal diploid: each autosome class (1–22) has exactly 2 chromosomes.
        self.max_autosome_count = 2
        # Female (XX) → 2; Male (XY) → 1.  We allow up to 2 for safety.
        self.max_x_count = 2
        # Y chromosome: at most 1 copy.
        self.max_y_count = 1


class TrainerConfig:
    def __init__(self):
        self.min_data_size_to_learn = 0
        self.cleaning_processes = 3
        self.vram_frac = 1.0
        self.batch_size = 256
        self.epoch_to_checkpoint = 1
        self.dataset_size = 50000
        self.start_total_steps = 0
        self.save_model_steps = 25
        self.load_data_steps = 100
        # [policy_loss_weight, value_loss_weight]
        self.loss_weights = [1.0, 1.0]
        self.learning_rate = 1e-4


class ModelConfig:
    # ── Input dimensions ─────────────────────────────────────────────────────
    # Maximum number of chromosomes per karyotype image (human: 46)
    n_chromosomes = 46
    # Number of chromosome classes (chromosomes 1–22, X, Y → 24 classes)
    n_classes = 24
    # Dimension of the per-chromosome visual embedding extracted by Mask2Former
    # backbone (ResNet-CNSN feature-map pooled to this size).
    embedding_dim = 256

    # ── Network architecture ──────────────────────────────────────────────────
    # 1-D ResNet over the chromosome sequence
    cnn_filter_num = 256
    cnn_first_filter_size = 3
    cnn_filter_size = 3
    res_layer_num = 5
    l2_reg = 1e-4
    value_fc_size = 256

    # ── Action space size ────────────────────────────────────────────────────
    # n_chromosomes × n_classes reassignment actions + 1 STOP action
    # 46 × 24 + 1 = 1105
    @classmethod
    def n_actions(cls):
        return cls.n_chromosomes * cls.n_classes + 1

    # Convenience index for the STOP action
    @classmethod
    def stop_action_index(cls):
        return cls.n_chromosomes * cls.n_classes

    distributed = False
