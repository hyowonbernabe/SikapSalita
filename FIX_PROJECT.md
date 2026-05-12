# Sikap-Salita — Setup & Fixes Guide

Everything needed to get Sikap-Salita running from a fresh clone. Apply these fixes before launching.

---

## Requirements

- Python **3.11.x** (not 3.12+, not 3.14 — PyTorch and MediaPipe wheels only exist for 3.11)
- Git (with Git LFS if models are not included in the repo)
- Windows, macOS, or Linux

---

## Step 1 — Verify Model Files Are Real

The `.pt` model files may be Git LFS pointers (134-byte stubs) instead of actual binaries.
Check file sizes after cloning:

```bash
# Should be tens or hundreds of MB — NOT 134 bytes
ls -lh trained_models/transformer/FSL105_classification/SignTransformer_best.pt
ls -lh trained_models/transformer/FSL105_ctc/SignTransformerCtc_best.pt
ls -lh trained_models/iv3_gru/FSL105_classification/InceptionV3GRU_best.pt
ls -lh trained_models/iv3_gru/FSL105_ctc/InceptionV3GRUCtc_best.pt
ls -lh trained_models/mediapipe_gru/FSL105_classification/MediaPipeGRU_best.pt
```

If files are tiny (< 1 KB), pull the real binaries via Git LFS:

```bash
git lfs install
git lfs pull --include="trained_models/**/*.pt"
```

---

## Step 2 — Create a Python 3.11 Virtual Environment

```bash
# Windows
py -3.11 -m venv .venv

# macOS / Linux
python3.11 -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

## Step 3 — Install PyTorch (CPU)

PyTorch requires a special index URL — do this **before** installing `requirements.txt`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you have an NVIDIA GPU and want CUDA support, find the right command at https://pytorch.org/get-started/locally/

---

## Step 4 — Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

`seaborn` is used by the app but missing from `requirements.txt`. Install it manually:

```bash
pip install seaborn
```

---

## Step 5 — Fix Model Path Resolution (Code Fix)

The app uses relative paths to find model checkpoints. These break if you launch Streamlit from any directory other than the project root. The fix is to make all paths absolute in `streamlit_app/core/config.py`.

Open `streamlit_app/core/config.py` and add this line right after the existing imports (after `from pathlib import Path`):

```python
_PROJECT_ROOT = Path(__file__).parent.parent.parent
```

Then update every `checkpoint_path` value in `MODEL_CONFIG` to use it:

```python
# BEFORE (breaks if CWD is wrong):
'checkpoint_path': 'trained_models/transformer/FSL105_classification/SignTransformer_best.pt',

# AFTER (works from any directory):
'checkpoint_path': str(_PROJECT_ROOT / 'trained_models/transformer/FSL105_classification/SignTransformer_best.pt'),
```

Apply this change to all six `checkpoint_path` entries:
- `transformer_isolated`
- `transformer_continuous`
- `iv3_gru_isolated`
- `iv3_gru_continuous`
- `mediapipe_gru_isolated`
- `mediapipe_gru_continuous`

Also update the page icon path:
```python
# BEFORE:
'page_icon': 'Sikap-Salita Icon.svg',

# AFTER:
'page_icon': str(_PROJECT_ROOT / 'Sikap-Salita Icon.svg'),
```

---

## Step 6 — Launch the App

```bash
streamlit run run_app.py
```

The app opens at **http://localhost:8501**.

To use a specific port:
```bash
streamlit run run_app.py --server.port 8080
```

---

## Verifying It Works

1. Open the app in your browser
2. In the sidebar under **RECOGNITION MODE**, select **Isolated Sign Recognition**
3. Under **MODEL STATUS**, both Transformer and InceptionV3-GRU should show as available (green)
4. Go to **Upload** → try one of the demo files from `data/demo/` (e.g. `clip_0138_nice to meet you.npz`)
5. Run a prediction — you should get a sign label with confidence score

---

## Summary of All Issues Found

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'seaborn'` | Missing from `requirements.txt` | `pip install seaborn` |
| "No Models Available" in sidebar | Relative model paths resolved against wrong CWD | Add `_PROJECT_ROOT` to `config.py`, make all checkpoint paths absolute |
| Model `.pt` files are 134 bytes | Git LFS pointers not downloaded | `git lfs pull --include="trained_models/**/*.pt"` |
| App crashes on Python 3.12+ | No PyTorch / MediaPipe wheels for 3.12+ | Use Python 3.11 venv |
