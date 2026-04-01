"""
Dual-head policy + value neural network for the karyotype correction agent.

Architecture:

Input
    Tensor of shape (1, STATE_DIM) = (1, 1198).
    Components (flattened in the trailing STATE_DIM dimension):
      · Predicted class probabilities (frozen, from Mask2Former):
          46 chromosomes × 24 classes = 1104 floats
      · Normalised hard assignments: 46 floats
      · Normalised class counts: 24 floats
      · Binary constraint violation indicators: 24 floats

Backbone
    Shared MLP: Dense(512, relu) → Dense(256, relu) → Dense(256, relu)

Policy head  (p)
    Dense(256, relu) → Dense(N_ACTIONS, softmax)
    N_ACTIONS = 46 × 24 + 1 = 1105

Value head   (v)
    Dense(value_fc_size, relu) → Dense(1, tanh)
    Outputs a scalar in [-1, +1] estimating the expected final reward.
"""

import hashlib
import json
import os
from logging import getLogger

import numpy as np

logger = getLogger(__name__)


class KaryotypeModel:
    """
    Policy + value network for the chromosome karyotype correction agent.

    Attributes
    ----------
    config : Config
        Global configuration (uses config.model which is a ModelConfig from
        configs/karyotype.py).
    model : keras.Model or None
        The Keras model after build() or load() is called.
    digest : str or None
        SHA-256 hash of the weight file currently loaded.
    api : KaryotypeModelAPI or None
        Pipe-based API for multi-process predictions.
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.digest = None
        self.api = None

    # ── Model lifecycle ───────────────────────────────────────────────────────

    def get_pipes(self, num: int = 1):
        """Return a list of ``num`` pipe connections to the prediction API."""
        if self.api is None:
            from chess_zero.agent.api_karyotype import KaryotypeModelAPI
            self.api = KaryotypeModelAPI(self)
            self.api.start()
        return [self.api.create_pipe() for _ in range(num)]

    def build(self):
        """Construct the Keras MLP model and store it in ``self.model``."""
        from keras.layers import Input, Dense, BatchNormalization, Activation
        from keras.models import Model
        from keras.regularizers import l2

        mc = self.config.model
        state_dim = mc.state_dim()     # 1198
        n_actions = mc.n_actions()     # 1105

        in_x = x = Input(shape=(1, state_dim), name='karyotype_input')

        # ── Shared MLP backbone ───────────────────────────────────────────────
        for i, units in enumerate(mc.hidden_units):
            x = Dense(
                units,
                kernel_regularizer=l2(mc.l2_reg),
                name=f'hidden_{i + 1}_dense{units}',
            )(x)
            x = BatchNormalization(name=f'hidden_{i + 1}_bn')(x)
            x = Activation('relu', name=f'hidden_{i + 1}_relu')(x)

        # ── Policy head ───────────────────────────────────────────────────────
        p = Dense(mc.hidden_units[-1], kernel_regularizer=l2(mc.l2_reg),
                  activation='relu', name='policy_dense1')(x)
        policy_out = Dense(
            n_actions,
            kernel_regularizer=l2(mc.l2_reg),
            activation='softmax',
            name='policy_out',
        )(p)

        # ── Value head ────────────────────────────────────────────────────────
        v = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg),
                  activation='relu', name='value_dense1')(x)
        value_out = Dense(
            1,
            kernel_regularizer=l2(mc.l2_reg),
            activation='tanh',
            name='value_out',
        )(v)

        self.model = Model(in_x, [policy_out, value_out],
                           name='karyotype_model')
        logger.debug(f"Built KaryotypeModel: input={in_x.shape} "
                     f"policy={policy_out.shape} value={value_out.shape}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self, config_path: str, weight_path: str) -> bool:
        """
        Load the model architecture from ``config_path`` (JSON) and weights
        from ``weight_path`` (HDF5).

        Returns True on success, False if the files do not exist or if the
        saved model's input shape does not match the current configuration
        (which triggers a rebuild by the caller).
        """
        from keras.models import Model

        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug(f"Loading KaryotypeModel from {config_path}")
            with open(config_path, 'rt') as f:
                loaded_model = Model.from_config(json.load(f))

            # Validate that the saved model's input shape matches the current
            # config.  A mismatch means the files are stale (built under a
            # different architecture) and the model must be rebuilt.
            expected_input_shape = (None, 1, self.config.model.state_dim())
            actual_input_shape = tuple(loaded_model.input_shape)
            if actual_input_shape != expected_input_shape:
                logger.warning(
                    f"KaryotypeModel input-shape mismatch: "
                    f"expected {expected_input_shape}, "
                    f"saved model has {actual_input_shape}. "
                    f"Discarding stale model and rebuilding."
                )
                return False

            self.model = loaded_model
            self.model.load_weights(weight_path)
            self.model._make_predict_function()
            self.digest = self.fetch_digest(weight_path)
            logger.debug(f"Loaded KaryotypeModel digest={self.digest}")
            return True
        logger.debug(
            f"KaryotypeModel files not found at {config_path} / {weight_path}")
        return False

    def save(self, config_path: str, weight_path: str):
        """
        Save the model architecture to ``config_path`` and weights to
        ``weight_path``.
        """
        logger.debug(f"Saving KaryotypeModel to {config_path}")
        with open(config_path, 'wt') as f:
            json.dump(self.model.get_config(), f)
        self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"Saved KaryotypeModel digest={self.digest}")

    @staticmethod
    def fetch_digest(weight_path: str) -> str:
        """Return the SHA-256 hash of a weight file."""
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, 'rb') as f:
                m.update(f.read())
            return m.hexdigest()
        return ''
