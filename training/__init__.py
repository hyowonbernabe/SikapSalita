"""
Training module for the Sikap-Salita Filipino Sign Language Recognition system.

This module provides training functionality for sign language recognition models,
including multi-task learning (gloss and category classification) with configurable loss weights.
Features are extracted via MediaPipe Holistic (75 landmarks, 225 values/frame) and consumed by:
- Siformer (SignTransformer): attention-based keypoint encoder
- Bi-LSTM baseline (InceptionV3GRU): recurrent model for temporal sequence modeling

Key Components:
- FSLDataset: PyTorch Dataset for sign language sequences
- FSLFeatureFileDataset: Dataset for precomputed MediaPipe Holistic features
- Training utilities and evaluation functions
- Support for both Siformer (Transformer) and Bi-LSTM baseline models

Usage:
    from training import FSLDataset, evaluate
    python -m training.train --model transformer --epochs 30
"""

# Conditional imports to avoid errors when dependencies are missing
try:
    from .utils import FSLDataset, evaluate
    TRAINING_UTILS_AVAILABLE = True
except ImportError:
    TRAINING_UTILS_AVAILABLE = False
    FSLDataset = None
    evaluate = None

try:
    from .train import (
        FSLFeatureFileDataset,
        train_model
    )
    TRAINING_TRAIN_AVAILABLE = True
except ImportError:
    TRAINING_TRAIN_AVAILABLE = False
    FSLFeatureFileDataset = None
    train_model = None

# Build __all__ list dynamically based on what's available
__all__ = []

if TRAINING_UTILS_AVAILABLE:
    __all__.extend(['FSLDataset', 'evaluate'])

if TRAINING_TRAIN_AVAILABLE:
    __all__.extend(['FSLFeatureFileDataset', 'train_model'])
