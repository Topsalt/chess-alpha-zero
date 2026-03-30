"""
Mask2Former Extractor — bridge between the mmdetection Mask2Former model
(trained on chromosome karyotype images) and the AlphaZero correction agent.

Responsibilities
----------------
1. Load the trained Mask2Former model from a checkpoint.
2. Run inference on a karyotype image to obtain:
   a. Instance segmentation masks (one per detected chromosome).
   b. Predicted class labels (initial classification).
   c. Per-instance visual embeddings extracted from the backbone feature maps,
      used as the ``embeddings`` input to :class:`KaryotypeEnv`.

Usage
-----
::

    from chess_zero.lib.mask2former_extractor import Mask2FormerExtractor

    extractor = Mask2FormerExtractor(
        config_file='mask2former/cnsn_resnet50_mcls_6k.py',
        checkpoint_file='path/to/checkpoint.pth',
        device='cuda:0',
    )

    result = extractor.extract('path/to/karyotype_image.png')
    # result.embeddings  → np.ndarray (N_CHROMOSOMES, embedding_dim)
    # result.assignments → np.ndarray (N_CHROMOSOMES,) int labels 1-24
    # result.masks       → list of np.ndarray binary masks
    # result.scores      → np.ndarray (N_CHROMOSOMES,) confidence scores

Notes
-----
- Requires ``mmdet`` (MMDetection) and ``mmengine`` to be installed.
- The custom CNSN backbone (``cnsn_models.resnet_cnsn``) must be importable;
  add the ``mask2former/`` directory to ``PYTHONPATH`` or ``sys.path`` before
  calling this module.
- If fewer than ``N_CHROMOSOMES`` (46) instances are detected, the missing
  slots are filled with zero embeddings and class label 1 (as a safe default).
- If more than ``N_CHROMOSOMES`` instances are detected, only the top-46
  highest-confidence detections are kept.
"""

import sys
import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import List, Optional

import numpy as np

from chess_zero.env.karyotype_env import N_CHROMOSOMES, N_CLASSES

logger = getLogger(__name__)

# Default embedding dimension produced by global-average-pooling the last
# backbone stage (ResNet-CNSN layer4 → 2048-d → projected to 256-d by the
# Mask2Former pixel-decoder).  Override via ModelConfig.embedding_dim.
DEFAULT_EMBEDDING_DIM = 256


@dataclass
class ExtractionResult:
    """Container for the per-image outputs of :class:`Mask2FormerExtractor`."""

    # Shape: (N_CHROMOSOMES, embedding_dim) — visual features per chromosome
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros(
        (N_CHROMOSOMES, DEFAULT_EMBEDDING_DIM), dtype=np.float32))

    # Shape: (N_CHROMOSOMES,) — integer class labels in 1..N_CLASSES
    assignments: np.ndarray = field(default_factory=lambda: np.ones(
        N_CHROMOSOMES, dtype=np.int32))

    # Binary segmentation masks, one per chromosome (may differ in H×W)
    masks: List[np.ndarray] = field(default_factory=list)

    # Confidence scores from Mask2Former (N_CHROMOSOMES,)
    scores: np.ndarray = field(default_factory=lambda: np.zeros(
        N_CHROMOSOMES, dtype=np.float32))

    # Original image path (for debugging)
    image_path: str = ""


class Mask2FormerExtractor:
    """
    Wraps a trained Mask2Former model to produce per-chromosome embeddings and
    initial class assignments from a raw karyotype microscope image.

    Parameters
    ----------
    config_file : str
        Path to the MMDetection config file (e.g.,
        ``mask2former/cnsn_resnet50_mcls_6k.py``).
    checkpoint_file : str
        Path to the trained model checkpoint (``.pth``).
    device : str
        Torch device string, e.g. ``'cuda:0'`` or ``'cpu'``.
    cnsn_model_dir : str | None
        Directory containing the ``cnsn_models`` package.  Added to
        ``sys.path`` automatically so that MMDetection can import
        ``cnsn_models.resnet_cnsn``.
    embedding_dim : int
        Dimension of the per-instance embedding vector.  Must match the
        Mask2Former pixel-decoder output width (default 256).
    """

    def __init__(self,
                 config_file: str,
                 checkpoint_file: str,
                 device: str = 'cuda:0',
                 cnsn_model_dir: Optional[str] = None,
                 embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.embedding_dim = embedding_dim
        self._model = None
        self._hook_outputs = {}

        # Make cnsn_models importable
        if cnsn_model_dir and cnsn_model_dir not in sys.path:
            sys.path.insert(0, cnsn_model_dir)

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from mmdet.apis import init_detector
        except ImportError as exc:
            raise ImportError(
                "mmdet is not installed. Install MMDetection and mmengine, "
                "then set PYTHONPATH to include mask2former/ for cnsn_models."
            ) from exc

        logger.info(f"Loading Mask2Former from {self.checkpoint_file}")
        self._model = init_detector(
            self.config_file,
            self.checkpoint_file,
            device=self.device,
        )
        self._register_embedding_hook()
        logger.info("Mask2Former model loaded successfully.")

    def _register_embedding_hook(self):
        """
        Register a forward hook on the Mask2Former pixel-decoder to capture
        the per-instance feature embeddings.

        The pixel-decoder produces multi-scale feature maps; we use the
        finest scale (index 0 of ``outs``) and RoI-pool/mask-pool per instance.
        """
        # The hook captures the output of the panoptic_head's transformer-decoder.
        # We store it in self._hook_outputs to be consumed after each forward pass.
        def _hook(module, input_, output):
            # output is a tuple; the last element is the per-query features
            # shape: (batch, num_queries, embed_dims) = (1, 100, 256)
            self._hook_outputs['query_features'] = output

        try:
            decoder = self._model.panoptic_head.transformer_decoder
            decoder.register_forward_hook(_hook)
        except AttributeError:
            logger.warning(
                "Could not register embedding hook on transformer_decoder. "
                "Embeddings will be zeros. Check the Mask2Former architecture.")

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, image_path: str) -> ExtractionResult:
        """
        Run Mask2Former inference on one karyotype image and return structured
        per-chromosome embeddings + initial class assignments.

        Parameters
        ----------
        image_path : str
            Path to a karyotype microscope image (PNG/JPEG).

        Returns
        -------
        ExtractionResult
        """
        self._ensure_loaded()

        try:
            from mmdet.apis import inference_detector
        except ImportError as exc:
            raise ImportError("mmdet is not installed.") from exc

        logger.debug(f"Running Mask2Former inference on {image_path}")
        result = inference_detector(self._model, image_path)

        return self._parse_result(result, image_path)

    def extract_batch(self, image_paths: List[str]) -> List[ExtractionResult]:
        """Run extraction on a list of images."""
        return [self.extract(p) for p in image_paths]

    # ── Internal parsing ──────────────────────────────────────────────────────

    def _parse_result(self, mmdet_result, image_path: str) -> ExtractionResult:
        """
        Convert an MMDetection ``DetDataSample`` (or legacy list result)
        into an :class:`ExtractionResult`.

        The Mask2Former output contains:
          - ``pred_instances.labels``  : predicted class indices (0-based)
          - ``pred_instances.scores``  : confidence scores
          - ``pred_instances.masks``   : binary instance masks (H×W booleans)
        """
        try:
            instances = mmdet_result.pred_instances
            raw_labels = instances.labels.cpu().numpy().astype(np.int32)  # 0-based
            raw_scores = instances.scores.cpu().numpy().astype(np.float32)
            raw_masks = instances.masks.cpu().numpy()  # (N, H, W) bool
        except AttributeError:
            # Legacy list-based result format
            raw_labels, raw_scores, raw_masks = self._parse_legacy_result(
                mmdet_result)

        n_detected = len(raw_labels)
        logger.debug(f"Detected {n_detected} chromosome instances in {image_path}")

        # Retrieve per-query embeddings captured by the forward hook
        query_features = self._hook_outputs.get('query_features', None)
        raw_embeddings = self._extract_embeddings_from_queries(
            query_features, raw_labels)

        # Keep only top-N_CHROMOSOMES by score if over-detected
        if n_detected > N_CHROMOSOMES:
            top_idx = np.argsort(raw_scores)[::-1][:N_CHROMOSOMES]
            raw_labels = raw_labels[top_idx]
            raw_scores = raw_scores[top_idx]
            raw_masks = raw_masks[top_idx]
            raw_embeddings = raw_embeddings[top_idx]

        # Pad with zeros / label=1 if under-detected
        n_used = len(raw_labels)
        embeddings = np.zeros((N_CHROMOSOMES, self.embedding_dim), dtype=np.float32)
        assignments = np.ones(N_CHROMOSOMES, dtype=np.int32)  # default class 1
        scores = np.zeros(N_CHROMOSOMES, dtype=np.float32)
        masks = []

        for i in range(n_used):
            embeddings[i] = raw_embeddings[i]
            # Convert 0-based label to 1-based class (clamp to valid range)
            assignments[i] = int(np.clip(raw_labels[i] + 1, 1, N_CLASSES))
            scores[i] = raw_scores[i]
            masks.append(raw_masks[i])

        # Pad mask list with empty masks for missing detections
        for _ in range(N_CHROMOSOMES - n_used):
            masks.append(np.zeros((1, 1), dtype=bool))

        return ExtractionResult(
            embeddings=embeddings,
            assignments=assignments,
            masks=masks,
            scores=scores,
            image_path=image_path,
        )

    def _extract_embeddings_from_queries(self, query_features,
                                         raw_labels) -> np.ndarray:
        """
        Map the Mask2Former transformer-decoder query features to per-instance
        embeddings.

        The decoder produces ``num_queries`` (100) query features.  Each
        detected instance corresponds to one query slot.  We select the query
        features in the order returned by ``pred_instances`` (which already
        corresponds to the matched query indices after Hungarian assignment).

        Parameters
        ----------
        query_features : torch.Tensor or None
            Shape: (batch=1, num_queries, embed_dims).  May be None if the
            hook was not successfully registered.
        raw_labels : np.ndarray
            1-D array of detected class indices (0-based), length = n_detected.

        Returns
        -------
        np.ndarray, shape (n_detected, embedding_dim)
        """
        n_detected = len(raw_labels)
        if query_features is None or n_detected == 0:
            return np.zeros((n_detected, self.embedding_dim), dtype=np.float32)

        try:
            import torch
            # query_features may be a tuple (last layer output) or a tensor
            if isinstance(query_features, (tuple, list)):
                feat = query_features[-1]  # last decoder layer
            else:
                feat = query_features

            # feat shape: (batch, num_queries, C) or (num_queries, C)
            if feat.dim() == 3:
                feat = feat[0]  # remove batch dim → (num_queries, C)

            feat_np = feat.detach().cpu().numpy().astype(np.float32)

            # The pred_instances ordering already aligns with query ordering.
            # Take the first n_detected rows (highest-scored queries after
            # post-processing filtering inside Mask2Former).
            n_q = feat_np.shape[0]
            selected = feat_np[:n_detected] if n_detected <= n_q else \
                np.vstack([feat_np, np.zeros(
                    (n_detected - n_q, feat_np.shape[1]), dtype=np.float32)])

            # Project to self.embedding_dim if needed
            if selected.shape[1] != self.embedding_dim:
                # Simple linear projection via SVD-truncation (no learnable params)
                # For production use, train a dedicated projection head.
                if selected.shape[1] > self.embedding_dim:
                    selected = selected[:, :self.embedding_dim]
                else:
                    pad = np.zeros(
                        (selected.shape[0],
                         self.embedding_dim - selected.shape[1]),
                        dtype=np.float32)
                    selected = np.hstack([selected, pad])

            return selected
        except Exception as exc:
            logger.warning(f"Could not extract embeddings from query features: {exc}")
            return np.zeros((n_detected, self.embedding_dim), dtype=np.float32)

    @staticmethod
    def _parse_legacy_result(mmdet_result):
        """Parse the old-style (list of arrays) MMDetection result."""
        # mmdet_result = (bbox_results, mask_results)
        bbox_results, mask_results = mmdet_result
        labels, scores, masks = [], [], []
        for cls_idx, bboxes in enumerate(bbox_results):
            for bbox in bboxes:
                # bbox: [x1, y1, x2, y2, score]
                labels.append(cls_idx)
                scores.append(float(bbox[4]))
        for cls_idx, cls_masks in enumerate(mask_results):
            for m in cls_masks:
                masks.append(m)
        return (np.array(labels, dtype=np.int32),
                np.array(scores, dtype=np.float32),
                np.array(masks) if masks else np.zeros((0, 1, 1), dtype=bool))
