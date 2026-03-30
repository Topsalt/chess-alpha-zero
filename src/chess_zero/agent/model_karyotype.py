"""
Dual-head policy + value neural network for the karyotype correction agent.

Architecture (analogous to model_chess.py):

Input
    Tensor of shape (N_CHROMOSOMES, embedding_dim + N_CLASSES)
    = (46, 256 + 24) = (46, 280) per default config.
    Each row encodes one chromosome: its visual embedding concatenated with
    a one-hot vector of its current class assignment.

Backbone
    1-D ResNet over the chromosome "sequence":
    Conv1D → BatchNorm → ReLU → [Residual blocks] → GlobalAveragePool

Policy head  (p)
    Dense → softmax over N_CHROMOSOMES × N_CLASSES + 1 = 1105 actions.

Value head   (v)
    Dense → tanh scalar in [-1, +1], estimating expected final reward.
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
        """Construct the Keras model and store it in ``self.model``."""
        # Import here so that Keras is not a hard dependency at import time
        from keras.layers import (Input, Conv1D, BatchNormalization,
                                   Activation, Add, Flatten, Dense,
                                   GlobalAveragePooling1D)
        from keras.models import Model
        from keras.regularizers import l2

        mc = self.config.model
        n_chr = mc.n_chromosomes           # 46
        n_cls = mc.n_classes               # 24
        emb_dim = mc.embedding_dim         # 256
        input_dim = emb_dim + n_cls        # 280
        n_actions = mc.n_actions()         # 1105

        in_x = x = Input(shape=(n_chr, input_dim), name='karyotype_input')

        # ── Backbone: 1-D ResNet ──────────────────────────────────────────────
        x = Conv1D(
            filters=mc.cnn_filter_num,
            kernel_size=mc.cnn_first_filter_size,
            padding='same',
            use_bias=False,
            kernel_regularizer=l2(mc.l2_reg),
            name=f'input_conv1d-{mc.cnn_first_filter_size}-{mc.cnn_filter_num}',
        )(x)
        x = BatchNormalization(name='input_batchnorm')(x)
        x = Activation('relu', name='input_relu')(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1, mc)

        # Global average pooling over the chromosome axis → (batch, filters)
        res_out = GlobalAveragePooling1D(name='global_avg_pool')(x)

        # ── Policy head ───────────────────────────────────────────────────────
        p = Dense(mc.cnn_filter_num, kernel_regularizer=l2(mc.l2_reg),
                  activation='relu', name='policy_dense1')(res_out)
        policy_out = Dense(
            n_actions,
            kernel_regularizer=l2(mc.l2_reg),
            activation='softmax',
            name='policy_out',
        )(p)

        # ── Value head ────────────────────────────────────────────────────────
        v = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg),
                  activation='relu', name='value_dense1')(res_out)
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

    @staticmethod
    def _build_residual_block(x, index: int, mc):
        """
        Build one 1-D residual block analogous to _build_residual_block in
        model_chess.py.
        """
        from keras.layers import (Conv1D, BatchNormalization, Activation, Add)
        from keras.regularizers import l2

        name = f'res{index}'
        in_x = x
        x = Conv1D(
            filters=mc.cnn_filter_num,
            kernel_size=mc.cnn_filter_size,
            padding='same',
            use_bias=False,
            kernel_regularizer=l2(mc.l2_reg),
            name=f'{name}_conv1-{mc.cnn_filter_size}-{mc.cnn_filter_num}',
        )(x)
        x = BatchNormalization(name=f'{name}_batchnorm1')(x)
        x = Activation('relu', name=f'{name}_relu1')(x)
        x = Conv1D(
            filters=mc.cnn_filter_num,
            kernel_size=mc.cnn_filter_size,
            padding='same',
            use_bias=False,
            kernel_regularizer=l2(mc.l2_reg),
            name=f'{name}_conv2-{mc.cnn_filter_size}-{mc.cnn_filter_num}',
        )(x)
        x = BatchNormalization(name=f'{name}_batchnorm2')(x)
        x = Add(name=f'{name}_add')([in_x, x])
        x = Activation('relu', name=f'{name}_relu2')(x)
        return x

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self, config_path: str, weight_path: str) -> bool:
        """
        Load the model architecture from ``config_path`` (JSON) and weights
        from ``weight_path`` (HDF5).

        Returns True on success, False if the files do not exist.
        """
        from keras.models import Model

        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug(f"Loading KaryotypeModel from {config_path}")
            with open(config_path, 'rt') as f:
                self.model = Model.from_config(json.load(f))
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
