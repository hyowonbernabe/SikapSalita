Bi-LSTM Baseline (InceptionV3-GRU) Model Directory
===================================================

This directory contains trained Bi-LSTM baseline (InceptionV3-GRU) model checkpoints for Sikap-Salita Filipino Sign Language Recognition. Features are extracted via MediaPipe Holistic — no CNN backbone is used at inference time.

Directory Structure
-------------------
Place your trained Bi-LSTM baseline (InceptionV3-GRU) models in the following structure:

iv3_gru/
├── FSL105_classification/
│   └── InceptionV3GRU_best.pt
└── FSL105_ctc/
    └── InceptionV3GRUCtc_best.pt

Required Files
--------------
1. Classification Model (Isolated Sign Recognition):
   - Path: FSL105_classification/InceptionV3GRU_best.pt
   - Used for: Isolated sign classification
   - Input: MediaPipe Holistic features [T, 2048]

2. CTC Model (Continuous Sign Recognition):
   - Path: FSL105_ctc/InceptionV3GRUCtc_best.pt
   - Used for: Continuous sign sequence recognition
   - Input: MediaPipe Holistic features [T, 2048]

Usage
-----
The Streamlit app automatically loads these models from:
- streamlit_app/core/config.py

Model Configuration:
- Input dimension: 2048 (MediaPipe Holistic keypoint features)
- Gloss classes: 105
- Category classes: 10
- CTC classes: 106 (for continuous model)

To use these models:
1. Copy your trained .pt checkpoint files to the appropriate subdirectories
2. Ensure file names match exactly: InceptionV3GRU_best.pt and InceptionV3GRUCtc_best.pt
3. The app will automatically detect and load them

Note: Model checkpoints should be PyTorch .pt files containing the model state_dict.

