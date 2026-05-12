"""
Models module for the Sikap-Salita Filipino Sign Language Recognition system.

Provides neural network architectures for sign language recognition. Features are
extracted using MediaPipe Holistic (75 landmarks, 225 values/frame) and fed into:
- Siformer (SignTransformer): attention-based encoder for keypoint sequences
- Bi-LSTM baseline (InceptionV3GRU): recurrent model for temporal sequence modeling
- CTC variants of both for continuous sign recognition

Classification models: SignTransformer, MediaPipeGRU, InceptionV3GRU
CTC models: SignTransformerCtc, MediaPipeGRUCtc, InceptionV3GRUCtc

Usage:
    from models import SignTransformer, MediaPipeGRU, InceptionV3GRU

    model = SignTransformer(num_gloss=105, num_cat=10)
"""

from .transformer import SignTransformer, SignTransformerCtc, PositionalEncoding
from .mediapipe_gru import MediaPipeGRU, MediaPipeGRUCtc
from .iv3_gru import InceptionV3GRU, InceptionV3GRUCtc

__all__ = [
    'SignTransformer',
    'SignTransformerCtc',
    'MediaPipeGRU',
    'MediaPipeGRUCtc',
    'PositionalEncoding', 
    'InceptionV3GRU',
    'InceptionV3GRUCtc'
]
