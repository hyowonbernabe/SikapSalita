# Sikap-Salita Research Paper — Complete Breakdown (Layman's Terms)

---

## PART 1 — Title & Authors

**"Sikap-Salita: A General-Purpose Filipino Sign Language (FSL) to Text-to-Speech Communicator"**

- 7 authors, all from Saint Louis University, Baguio City
- "Sikap" = effort/gesture, "Salita" = word/speech

---

## PART 2 — Abstract (Quick Summary)

Whole paper squeezed into one paragraph. Plain terms:

- Filipino Sign Language (FSL) legally recognized in Philippines, but almost no tools to help deaf people communicate with hearing people
- Built **Sikap-Salita** — web app that watches someone sign FSL on camera, figures out what sign was made, translates it, speaks it out loud (in English or Filipino)
- Trained **two AI models** to recognize signs and compared which is better:
  - **Bi-LSTM** (simpler, older approach) — teaches computer to read hand movements one moment at a time
  - **Siformer** (newer, more advanced) — watches left hand, right hand, body separately, then combines everything
- Dataset used: **FSL-105** — 105 different FSL signs, 2,130 video clips
- Results:
  - Siformer: **96.2% accuracy**
  - Bi-LSTM: **91.7% accuracy**
  - Whole system goes from "sign recognized" to "voice speaking" in **1.8 seconds average**
  - Runs at **28–30 frames per second** (smooth, real-time)
  - **94.1% end-to-end accuracy** (correct sign → correct spoken word)

**Jargon breakdown:**
- **Bi-LSTM** = Bidirectional Long Short-Term Memory — AI that processes sequences both forward and backward in time
- **Siformer** = "Sign-Isolated Transformer" — AI that separates body parts into their own "lanes" before combining them
- **MediaPipe** = Google's tool that detects body landmarks (joints, fingers) in real-time
- **NLLB-200** = Meta's translation AI that handles 200 languages
- **Qwen3-TTS** = Alibaba's text-to-speech AI
- **MOS (Mean Opinion Score)** = standard 1-5 rating scale for audio quality

---

## PART 3 — Introduction

**Problem being solved:**
- 70 million deaf people worldwide face huge communication barriers
- In Philippines: 151,459 Filipinos deaf or hard of hearing (per DOH, Feb 2026)
- RA 11106 (Filipino Sign Language Act of 2018) made FSL official, but barely any tools exist
- Not enough FSL interpreters → deaf people struggle in hospitals, courts, schools, workplaces

**What this study aims to do (4 goals):**
1. Extract "skeleton" data (body joint positions) from FSL videos
2. Train two AI models and compare them fairly
3. Add translation (English → Filipino) and voice output
4. Build working app and measure how well it performs

**Why this matters:**
- First time Siformer architecture applied to FSL
- First time two models compared side-by-side under identical conditions on FSL
- First time neural TTS (AI voice) integrated with FSL recognition for Filipinos
- Aligns with UN goals: Quality Education, Reduced Inequalities, Sustainable Cities

---

## PART 4 — Review of Related Literature

Section looks at what other researchers already did.

**Other sign language systems (non-FSL):**
- ASL (American Sign Language): One system reached 82.9% accuracy, another 99.98% (for static alphabet only)

**FSL-specific systems (most relevant):**

| Study | What they did | Result |
|-------|---------------|--------|
| Pilare et al. (2024) | MediaPipe + basic ML | Recognized 3 words, 27 FSL phrases |
| Cayme et al. (2024) | CNN-LSTM | 15 FSL gestures, lightweight |
| Montefalcon et al. | LSTM | 94% on 15 continuous FSL phrases |
| Signify (2025) | LSTM vs Transformer on FSL-105 | Transformer: 98.73% accuracy |
| TinyFSL (2024) | Compressed model for mobile | Works on phones without internet |
| KamAI (2025) | CNN + MediaPipe on Android | 83–91% accuracy, users rated "Highly Acceptable" |

**Key gap identified:** Nobody used Siformer on FSL, nobody compared two models on FSL-105 under identical conditions, nobody added AI voice output to FSL system for Filipinos.

**Jargon:**
- **CNN** = Convolutional Neural Network — AI good at recognizing patterns in images
- **LSTM** = Long Short-Term Memory — AI good at processing sequences over time (like video)
- **Transformer** = More advanced AI that looks at all parts of sequence simultaneously
- **PICOC** = Population, Intervention, Comparison, Outcome, Context — structured way to do literature review
- **ISO 25010** = International standard for evaluating software quality

---

## PART 5 — Methodology

### 5.1 Overall Approach
Applied experimental research — build something, measure how well it works. Five phases:
1. Build data pipeline (extract skeleton data, clean it, multiply it)
2. Train and compare two AI models
3. Add TTS and translation
4. Build web app
5. Test everything end-to-end

### 5.2 Data Collection
Used **FSL-105 dataset** — ONLY publicly available labeled FSL video dataset that exists.
- 105 signs
- 2,130 video clips (~4 seconds each)
- Shot on plain blue background by one deaf adult FSL signer
- Reviewed by FSL expert
- Categories: Greeting, Survival, Number, Calendar, Days, Family, Relationships, Color, Food, Drink
- ~20 video clips per sign
- Pre-split: 80% training (~1,525 clips), 20% testing (~426 clips)

### 5.3 Data Preprocessing

Instead of feeding raw video pixels into AI, they convert each frame into **skeleton data** — just positions of 75 body landmarks (joints/fingertips). Smarter because:
- Not affected by background color, clothing, skin tone
- Much smaller data = faster processing

**Three steps:**

**Step 1 — Landmark Extraction:**
MediaPipe identifies 75 points per frame:
- 21 left hand landmarks
- 21 right hand landmarks
- 33 upper body pose points
- Each landmark has X, Y, Z coordinates = 225 numbers per frame
- Face NOT included (FSL signs use hands/arms, not face, in this dataset)
- If MediaPipe misses a hand in a frame, it copies last known position

**Step 2 — Normalization (making data consistent):**
- **Re-centering:** Subtract mid-chest position so signer's position in frame doesn't matter
- **Shoulder-width scaling:** Divide all positions by distance between shoulders so body size doesn't matter
- **Temporal resampling:** All clips squished or stretched to exactly 64 frames, regardless of original length

**Jargon:** Normalization = making all data same scale/units so AI doesn't get confused by irrelevant differences

**Step 3 — Data Augmentation (making fake variations of training data):**
Only 20 clips per sign is small. To simulate more variety, randomly alter training data each session:

| Technique | What it does | How often applied |
|-----------|-------------|-------------------|
| Time warp | Speed up or slow down random parts | 50% |
| Gaussian noise | Add tiny random jitter to coordinates | 50% |
| Horizontal mirror | Flip left/right (simulate left-handed signer) | 50% |
| Scale jitter | Make signer appear slightly bigger or smaller (0.85x–1.15x) | 30% |
| Frame drop | Remove 1-3 random frames then re-interpolate | 30% |
| Spatial shift | Move all landmarks slightly up/down/left/right | 30% |

**Jargon:** Augmentation = creating fake but realistic variations of training examples so AI learns to handle real-world differences

### 5.4 Sign Recognition Models

**Model A — Bi-LSTM (Baseline):**
Reader that goes through sign frame by frame, both forward and backward, to understand motion. Then "pays attention" to most important frames and picks from 105 possible signs.
- ~2.4 million parameters (settings AI learns)

**Jargon:**
- **Attention pooling** = mechanism that figures out which frames are most important
- **Dropout** = randomly turning off some neurons during training to prevent overfitting
- **Softmax** = converts raw scores into probabilities that add up to 100%
- **AdamW** = optimization algorithm (how AI adjusts itself during training)
- **Cosine decay** = gradually reducing learning rate in smooth curve over time
- **Label smoothing** = soften 100% confidence to prevent overconfidence

**Model B — Siformer (Advanced):**
Instead of treating all 225 features as one big blob, splits them into 3 separate "lanes":
- Left hand lane → its own transformer
- Right hand lane → its own transformer
- Body pose lane → its own transformer
Then fuses all three at the end.

Benefit: If one hand is hidden or missing, only that lane is affected. Other two still work correctly.
- ~7.2 million parameters
- Used pre-trained weights from WLASL100 (American Sign Language dataset) as starting point

**Jargon:**
- **Transfer learning** = using AI that already learned from one dataset as starting point for new dataset
- **WLASL100** = Dataset of 100 American Sign Language signs used to pre-train Siformer
- **Transformer streams** = separate processing pipelines for each body part
- **Fusion** = combining outputs of all three streams into final decision
- **Kinematic hand pose rectification** = correcting anatomically impossible hand poses

### 5.5 Text-to-Speech and Translation

Once sign recognized (e.g., "FRIEND"), pipeline:
1. **Translates English label → Filipino** using Facebook NLLB-200 (600M-parameter translation model)
   - Also has hand-made dictionary for all 105 FSL-105 labels as backup
2. **Converts text → speech** using Qwen3-TTS (0.6B model, outputs audio at 24kHz quality)
   - Filipino not officially supported by Qwen3-TTS, tested empirically

**Jargon:**
- **NLLB-200** = "No Language Left Behind" — Meta's AI that translates between 200 languages
- **600M distilled** = compressed version of larger model (600 million parameters)
- **tgl_Latn** = language code for Filipino (Tagalog) in Latin script
- **24kHz** = audio sample rate (CD quality is 44.1kHz; 24kHz still very clear)
- **Override dictionary** = manually written translations that take priority over AI

### 5.6 Evaluation Plan

Tested system at 3 levels:

1. **Model-level:** How accurately does each model recognize signs?
   - Metrics: Top-1 accuracy, Top-5 accuracy, Macro F1, confusion matrix
   - Also tested with 5-fold cross-validation

2. **TTS-level:** How good is voice output?
   - Latency (must be under 2 seconds), success rate, MOS ratings from 5-10 listeners, translation accuracy

3. **End-to-end:** Does whole chain work together?
   - Full accuracy, frames per second (must be 25+ fps), sign-to-speech latency, false trigger rate

**Jargon:**
- **Top-1 accuracy** = was correct answer the #1 guess? (strictest measure)
- **Top-5 accuracy** = was correct answer in top 5 guesses? (more lenient)
- **Macro F1** = average accuracy per class, treating all 105 signs equally
- **Confusion matrix** = table showing which signs got confused with which other signs
- **5-fold cross-validation** = split data into 5 groups, train on 4, test on 1, rotate 5 times
- **False trigger rate** = how often system thinks you signed something when you didn't
- **FPS (frames per second)** = how smooth video processing is (25+ = smooth enough for real-time)

---

## PART 6 — Results

### 6.1 Model Accuracy (80/20 Split)

| Metric | Bi-LSTM | Siformer |
|--------|---------|----------|
| Top-1 Accuracy | 91.7% | **96.2%** |
| Top-5 Accuracy | 97.3% | **99.1%** |
| Macro F1 | 0.908 | **0.955** |
| Weighted F1 | 0.919 | **0.961** |
| # of Parameters | ~2.4M | ~7.2M |

Siformer wins by **4.5 percentage points** on top-1 accuracy. For every 100 signs, Siformer gets ~4-5 more right than Bi-LSTM.

### 6.2 Five-Fold Cross-Validation

| Model | Mean Accuracy | Std Deviation |
|-------|--------------|---------------|
| Siformer | **94.8%** | 1.2% |
| Bi-LSTM | 89.9% | 1.8% |

Siformer not just more accurate — also more **consistent** (lower standard deviation). 4-5 point gap held across all 5 folds.

**Jargon:**
- **Standard deviation (SD)** = how much results vary. Lower SD = more reliable, consistent results.

### 6.3 Per-Category Performance

Easiest categories (both models did well):
- NUMBER signs: Siformer 99.0%, Bi-LSTM 96.8%
- CALENDAR: Siformer 98.1%, Bi-LSTM 95.2%
- DAYS: Siformer 97.6%, Bi-LSTM 94.3%

Hardest categories (signs that look similar to each other):
- **RELATIONSHIPS**: Siformer 92.4%, Bi-LSTM 85.1%
  - "FRIEND" vs "NEIGHBOR" — same hand path, only final hand position differs by few centimeters. Bi-LSTM can't focus on this detail. Siformer's separate hand streams catch it.
- **SURVIVAL**: Siformer 93.1%, Bi-LSTM 86.7%
  - "DEAF" vs "HARD OF HEARING" — both start with pointing at ear, but "DEAF" ends with definitive close, "HARD OF HEARING" ends with wavering motion. Bi-LSTM got this wrong **8 out of 18 times** (55.6% error rate for this pair).

### 6.4 Training Dynamics

- **Bi-LSTM**: Learned quickly, peaked at epoch 65, stopped at epoch 80
- **Siformer**: Slower start (3 streams needed to sync up), overtook Bi-LSTM at epoch 40, peaked at epoch 95, stopped at epoch 115

**Transfer learning mattered:** Without WLASL pre-training, Siformer only reached 93.4% (vs 96.2% with it). Pre-training on ASL signs gave foundational knowledge of hand shapes that transferred to FSL.

**Jargon:**
- **Epoch** = one complete pass through all training data
- **Early stopping** = stop training when performance stops improving, prevent overfitting
- **Cosine warmup** = starting with very small learning rate and gradually increasing it

### 6.5 Augmentation Ablation

Which augmentations helped most?

| Removed | Accuracy Drop |
|---------|--------------|
| Time warp removed | **-2.1%** (most important) |
| Gaussian noise removed | -1.7% |
| Horizontal mirror removed | -0.8% |
| Scale jitter removed | -0.6% |
| Frame drop removed | -0.5% |
| Spatial shift removed | -0.4% |

**Time warp** mattered most because different signers naturally sign at different speeds. **Gaussian noise** mattered second because real-world cameras don't give perfect landmark positions.

**Note on horizontal mirroring:** HELPED most signs but HURT 3 signs in SURVIVAL category that rely on one specific hand doing action to the other (handedness-dependent signs). Per-class flag system fixed this by disabling mirroring for those signs.

**Jargon:**
- **Ablation** = systematically removing one thing at a time to see how much it mattered

### 6.6 TTS and Translation Performance

| Metric | English | Filipino |
|--------|---------|----------|
| Mean latency | 1.2 seconds | 1.4 seconds |
| 95th percentile latency | 1.8 seconds | 2.1 seconds |
| Intelligibility success rate | **100%** (105/105) | 96.2% (101/105) |
| MOS (audio quality rating) | **4.1 / 5.0** | 3.4 / 5.0 |

Extra 0.2 seconds for Filipino = NLLB-200 translation time (150-250ms).

**4 problematic Filipino labels:** Words with repeated syllables (e.g., "kapatid") got unnatural stress patterns. Fix: feed phonetic spelling ("ka-pa-tid") to TTS.

**MOS evaluation setup:** 8 listeners — 4 native Filipino speakers, 4 English speakers — each rated 20 randomly selected labels.

Filipino TTS (3.4/5.0) lower because Qwen3-TTS NOT trained on Filipino specifically — handles it through general multilingual capability, works but sounds slightly unnatural.

### 6.7 End-to-End Performance

Testing setup: One evaluator performed all 105 signs **3 times each** = **315 total sign attempts**

- Frame rate: **28-30 FPS** ✓ (target was 25+)
- GPU usage: 62% (room to spare on RTX 5060)
- End-to-end accuracy: **94.1%** (296/315 correct)
- Mean sign-to-speech latency: **1.8 seconds**

19 errors broke down as:
- 11 misrecognitions (wrong sign predicted)
- 5 dwell timeout failures (sign too fast, system didn't commit)
- 3 false triggers (system thought non-sign movement was sign)
- **False trigger rate: 2.3%**

**Jargon:**
- **Dwell-based commitment** = system only "commits" to recognized sign after holding same prediction steady for set time (prevents jittery false triggers)
- **False trigger** = system recognizes something when you weren't signing (e.g., scratching face)

---

## PART 7 — Discussion

### Key takeaways from results:

**Model comparison:** Siformer's separate hand/body streams let it focus on subtle hand shape differences. Bi-LSTM sees all 225 features as equal — can't prioritize hand details over body movements.

**vs. Signify (98.73%):** Siformer 2.5 points behind Signify's Transformer. Why? Feature isolation trades some accuracy for robustness. In controlled FSL-105 dataset (one signer, clean background), standard full attention wins. In messy real-world conditions, Siformer's robustness would likely close gap.

**TTS gap (0.7 MOS points):** English sounds better because Qwen3-TTS trained on it specifically. Filipino works but sounds like non-native speaker reading Filipino text. Pre-recorded audio considered but rejected — wouldn't scale beyond 105 words.

**Latency design decision:** 67% of 1.8 second latency = 1.2 second dwell time. Intentional — shorter dwell = faster, but more false triggers. App lets users adjust dwell time.

**Word order problem:** Signs come out in FSL word order, not natural English/Filipino sentence order. For simple expressions fine. For complex sentences, sounds unnatural.

---

## PART 8 — Limitations

1. **Single signer dataset** — all training data is one person. Different signers may get lower accuracy
2. **Controlled background** — real-world messy backgrounds may cause problems
3. **Only 105 signs** — FSL has ~5,000+ signs. Covers tiny fraction
4. **Only 8 MOS listeners** — statistically too few for strong conclusion
5. **Single evaluator for end-to-end test** — not actual deaf FSL users signing naturally
6. **No live webcam at time of writing** — demo uses pre-recorded input; real-time webcam is future goal

---

## PART 9 — Conclusion

**What was built:** Sikap-Salita — web app that recognizes 105 FSL signs from video and speaks them aloud in English or Filipino.

**What was proven:**
- Siformer > Bi-LSTM for FSL (96.2% vs 91.7%)
- 1.8 second average sign-to-speech latency
- 94.1% end-to-end accuracy
- Runs in real-time (28-30 FPS) on consumer hardware (RTX 5060)

**Three "firsts" claimed:**
1. First Siformer applied to FSL
2. First controlled model comparison on FSL-105
3. First neural TTS integrated with FSL recognition for Filipinos

**Future directions:**
- More signers in dataset
- Expand beyond 105 signs
- Continuous sign recognition (remove dwell system)
- Mobile deployment
- Dedicated Filipino TTS model

---

# Consistency Fixes Applied

5 inconsistencies found and fixed in `SikapSalitaResearchPaper-Final.md`. Same edits must be applied to Google Docs original.

---

## Change 1 — Introduction, Goal #4

**Ctrl+F:** `develop a PyQt6 desktop application`

**Replace with:**
> develop a FastAPI web application

**Why:** Actual built system is FastAPI web app, not PyQt6 desktop. Abstract and conclusion already say "web-based" — this goal was stale.

---

## Change 2 — Section 2.1 Materials and Methods

**Ctrl+F:** `desktop application development; and (5) end-to-end`

**Replace with:**
> web application development; and (5) end-to-end

**Why:** Same as Change 1 — phase 4 description was stale.

---

## Change 3 — Section 2.2.3, Model B description

**Ctrl+F:** `Additional features include kinematic hand pose rectification and input-adaptive inference (~5–10M parameters, 86.50% top-1 accuracy on WLASL100).`

**Replace with:**
> Additional features include kinematic hand pose rectification and input-adaptive inference (~5–10M parameters). In its original paper [14], Siformer achieved 86.50% top-1 accuracy on WLASL100 (an ASL dataset); this study evaluates its performance on FSL-105 under identical preprocessing conditions.

**Why:** Original text said "86.50% on WLASL100" but readers might think that's the study's FSL result. Then Results say 96.2% — appears inconsistent. Clarification makes clear 86.50% is from original ASL paper, not this study.

---

## Change 4 — Table 3 (Model B Settings)

**Ctrl+F:** `0.001, with cosine decay` (inside Table 3 — verify right table)

**Replace these values:**

| Setting | Old Value | New Value |
|---------|-----------|-----------|
| Learning rate | `0.001, with cosine decay` | `0.0005, with cosine warmup + decay` |
| Batch size | `32` | `16` |
| Epochs | `100, with early stopping (patience 15)` | `150, with early stopping (patience 20)` |

*(Optimizer and Loss function rows stay the same)*

**Why:** Table 3 labeled "Model B Settings" but contained Model A (Bi-LSTM) values. Prose above Table 3 says Siformer uses LR 0.0005, batch 16, 150 epochs, patience 20 — those values now in table.

---

## Change 5 — Table 4 (System Architecture), last row

**Ctrl+F:** `Desktop App` (in the table row)

**Replace entire row with:**

| Web Application | FastAPI + Browser Frontend | CPU | Lightweight web server; enables real-time skeleton visualization and bilingual output in any browser |

**Why:** Same as Change 1 — stale PyQt6 reference. Replaced with actual stack.

---

## Saved location

`c:\Projects\Sikap Salita\docs\research\SikapSalitaResearchPaper-Breakdown.md`
