# Whisper Fine-Tuning Project

**Project:** Swiss German Speech-to-Text Fine-tuning  
**Organization:** i4Ds (Institute for Data Science, FHNW)  
**Repository:** [i4Ds/whisper-finetune](https://github.com/i4Ds/whisper-finetune)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Key Features](#key-features)
5. [Data Pipeline](#data-pipeline)
6. [Training Pipeline](#training-pipeline)
7. [Model Architecture](#model-architecture)
8. [Configuration System](#configuration-system)
9. [How to Use](#how-to-use)
10. [Deployment](#deployment)
11. [Technical Details](#technical-details)

---

## Project Overview

### Purpose
Fine-tune OpenAI's Whisper speech-to-text models for improved performance on Swiss German and German language transcription tasks. The project focuses on handling domain-specific audio (e.g., office environments, low-quality recordings) and dialectal variations.

### Key Objectives
- **Timestamp Training**: Maintain temporal alignment in transcriptions
- **Prompt Training**: Use contextual prompts to improve accuracy
- **Robust Augmentation**: Handle various audio quality conditions
- **Production Deployment**: Convert models to faster-whisper format for efficient inference

### Supported Models
- Whisper Large V3
- Whisper Large V3 Turbo
- Custom configurations for specific use cases

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                          │
│  HuggingFace Datasets → Filtering → Sampling → Augmentation     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Audio        │  │ SpecAugment  │  │ Mel          │          │
│  │ Augmentation │→ │ (Time/Freq)  │→ │ Spectrogram  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                                      │                 │
│         ▼                                      ▼                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │        Whisper Model (with Modifications)        │          │
│  │  - Stochastic Depth                              │          │
│  │  - Gradient Checkpointing                        │          │
│  │  - Mixed Precision Training                      │          │
│  │  - Deep SpecAugment (in encoder layers)          │          │
│  └──────────────────────────────────────────────────┘          │
│         │                                      │                 │
│         ▼                                      ▼                 │
│  ┌──────────────┐              ┌──────────────────────┐        │
│  │ WandB        │              │ Model Checkpoints    │        │
│  │ Logging      │              │ (best/last)          │        │
│  └──────────────┘              └──────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL CONVERSION                            │
│  OpenAI Format → HuggingFace → CTranslate2 (faster-whisper)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
whisper-finetune/
│
├── configs/                          # Training configuration files
│   ├── DEBUG.yaml                    # Debug/development config
│   ├── large-v3-sg-corpus-mc-*.yaml  # Large V3 model configs
│   └── turbo-v3-*.yaml               # Turbo V3 model configs
│
├── src/whisper_finetune/             # Main package
│   ├── __init__.py
│   ├── utils.py                      # General utilities (config, seeding, etc.)
│   │
│   ├── data/                         # Data processing module
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Dataset & DataLoader implementation
│   │   └── utils.py                  # Data utilities (augmentation, filtering)
│   │
│   ├── model/                        # Model-related code
│   │   ├── __init__.py
│   │   ├── model_utils.py            # Model loading, training, evaluation
│   │   ├── optimizer.py              # Optimizer configuration
│   │   ├── scheduler.py              # Learning rate schedulers
│   │   ├── augment.py                # Audio augmentation pipelines
│   │   ├── lora.py                   # LoRA utilities for parameter-efficient fine-tuning
│   │   └── bg_noise/                 # Background noise samples for augmentation
│   │
│   ├── eval/                         # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── wer.py                    # Word Error Rate implementation
│   │   └── utils.py                  # Text normalization, vocab specs
│   │
│   └── scripts/                      # Executable scripts
│       ├── __init__.py
│       ├── finetune.py               # Main training script
│       ├── convert_openai_to_hf.py   # OpenAI → HuggingFace conversion
│       ├── convert_c2t.py            # HuggingFace → CTranslate2 conversion
│       └── wandb_to_ct2_upload.py    # WandB → CT2 pipeline
│
├── whisper_v3_utils/                 # Whisper V3 tokenizer & config
│   ├── config.json
│   └── tokenizer.json
│
├── whisper_v3_turbo_utils/           # Whisper V3 Turbo tokenizer & config
│   ├── config.json
│   └── tokenizer.json
│
├── pyproject.toml                    # Project metadata & dependencies
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
│
├── sc_sbatch.sh                      # SLURM batch submission script
├── sc_debug.sh                       # SLURM debug script
├── multi_submit.sh                   # Multi-config batch submission
│
└── README.md                         # Project documentation
```

---

## Key Features

### 1. **Advanced Data Augmentation**

#### Audio Augmentation (`model/augment.py`)
- **Baseline Augmentation**:
  - Background noise injection (office sounds, ambient noise)
  - Gaussian noise / SNR-based noise
  - Loudness normalization
  - Aliasing effects
  - **TimeStretch**: Speed up/slow down audio without pitch change (0.75x-1.25x)

- **Advanced Augmentation** (formerly part of baseline):
  - Low/High/Band-pass filters
  - Gain and gain transitions
  - Pitch shifting
  - Audio shifting
  - Clipping distortion
  - Air absorption simulation
  
- **Office Environment Augmentation**:
  - Room simulation (carpeted office: 3-5m × 2.5-4m × 2.4-3m)
  - MP3 compression (8-64 kbps) - simulates low-quality recordings
  - Bit crushing (6-14 bits) - simulates cheap ADCs

#### Spectrogram Augmentation (`data/data_loader.py`, `data/utils.py`)
- **SpecAugment** (applied to mel-spectrograms):
  - Time masking (configurable mask parameter)
  - Frequency masking (configurable mask parameter)
  - Time warping (LibriSpeech-style implementation)
  
- **Deep SpecAugment** (`model/model_utils.py`):
  - Applied **inside encoder layers** (SpecAugment++ approach)
  - Operates on normalized features between attention blocks
  - More robust than input-only augmentation

- **Extremes Frequency Masking** (`data/utils.py`):
  - Masks low/high frequency extremes
  - Helps model focus on speech-relevant frequencies

#### Text Augmentation
- **BPE Dropout**: Stochastic tokenization for better generalization

### 2. **LoRA (Low-Rank Adaptation)**

Parameter-efficient fine-tuning using the [minLoRA](https://github.com/cccntu/minLoRA) library.

#### Features
- Apply LoRA to entire model, encoder-only, or decoder-only
- Configurable rank, alpha, and dropout
- Automatically freezes non-LoRA parameters
- Dramatically reduces trainable parameters (~0.1-1% of full model)

#### Configuration
```yaml
model:
  lora: true
  lora_config:
    rank: 16          # LoRA rank (lower = fewer params)
    lora_alpha: 32    # Scaling factor (typically 2x rank)
    lora_dropout: 0.1 # Dropout for regularization
```

#### LoRA + Training Mode Combinations
| `train_only_decoder` | `train_only_encoder` | LoRA Applied To |
|---------------------|---------------------|-----------------|
| `false` | `false` | Entire model |
| `true` | `false` | Decoder only |
| `false` | `true` | Encoder only |

### 3. **Timestamp & Prompt Training**

#### Timestamp Handling
- Preserves temporal information in transcriptions
- Format: `<|0.00|>` to `<|30.00|>` in 0.02s increments
- Configurable timestamp training rate
- Partial segment handling (cuts audio at last timestamp)

#### Prompt-Based Training
- Uses previous transcription context as prompt
- Configurable prompt length (max 223 tokens)
- Prompt use rate: 50% (configurable)
- Special token handling for prompt vs. transcription

### 3. **Training Optimizations**

#### Stochastic Depth
- Implementation from the Whisper paper
- Randomly skips layers during training
- Improves generalization and reduces overfitting
- Applied to both encoder and decoder

#### Gradient Checkpointing
- Trades compute for memory
- Separate options for encoder and decoder
- Enables larger batch sizes on limited GPU memory

#### Mixed Precision Training
- FP16 or BF16 support
- Gradient scaling for FP16 stability
- Conditional scheduler stepping based on grad scale

#### Memory Efficiency
- 8-bit optimizers (Adam8bit, AdamW8bit via bitsandbytes)
- Gradient accumulation
- BFloat16 model weights option

### 4. **Monitoring & Logging**
- **Weights & Biases Integration**:
  - Training/validation loss tracking
  - Word Error Rate (WER) monitoring
  - Learning rate logging
  - Gradient scale tracking
  - Model artifact storage

---

## Data Pipeline

### Data Sources
Configured via `dataset` section in YAML configs:

```yaml
dataset:
  train_datasets: 
    - i4ds/sg_corp_train_no_overlap_speaker_ret
    - i4ds/srg-full-train-val-v2
    - i4ds/mozilla-cv-13-long-text-de
  
  select_n_per_t_ds: [null, null, 15000]
  groupby_col: [null, null, null]
```

### Data Processing Flow

```
1. Load from HuggingFace Datasets
   ↓
2. Filter & Validate
   - Check audio array validity
   - Validate text (must be string)
   - Remove samples < 6 seconds or < 30 chars
   ↓
3. Sample (optional)
   - Random sampling: select N samples
   - Groupby sampling: select N per group (e.g., per speaker)
   ↓
4. Concatenate Multiple Datasets
   ↓
5. Apply Augmentations (during training)
   - Audio augmentation (baseline + office)
   - Mel spectrogram generation
   - SpecAugment (time/freq masking, warping)
   - Extreme frequency masking
   ↓
6. Tokenization
   - BPE with optional dropout
   - Timestamp encoding
   - Prompt construction
   ↓
7. Batch & Pad
   - Pad sequences to max length
   - Create decoder input/output tensors
```

### Data Format
**Input to model:**
- `x`: Mel spectrogram (batch, n_mels, n_frames)
  - n_mels: 80 (Whisper V1/V2) or 128 (Whisper V3)
  - n_frames: 3000 (30 seconds at 100 frames/sec)

- `y_in`: Decoder input tokens (batch, seq_len)
  - Format: `[sot_prev?] + [prompt?] + [sot, lang, task, no_timestamps?] + text_tokens`

- `y_out`: Decoder target tokens (batch, seq_len)
  - Format: `[-100 for prompts] + [special_tokens] + text_tokens + [eot]`
  - `-100` = ignore index for cross-entropy loss

---

## Training Pipeline

### Training Loop (`scripts/finetune.py`)

```python
1. Initialize Model
   - Load pre-trained Whisper weights
   - Apply modifications (stochastic depth, checkpointing)
   - Move to GPU
   
2. Setup Training Components
   - Datasets (train/val)
   - DataLoaders
   - Optimizer (Adam/AdamW, 8-bit optional)
   - LR Scheduler (linear, cosine, cosine with restarts)
   
3. Main Training Loop
   for step in range(1, train_steps + 1):
       a. Training Step
          - Gradient accumulation
          - Mixed precision forward pass
          - Loss computation (cross-entropy)
          - Backward pass with gradient clipping
          - Optimizer step
          
       b. Validation (periodic)
          - Evaluate on dev set
          - Compute WER
          - Save best model (lowest WER)
          - Log to WandB
          
       c. Checkpointing
          - Save model at intervals (optional)
          - Save final model
          
4. Cleanup
   - Log peak memory usage
   - Finish WandB run
```

### Key Training Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `batch_size` | Samples per batch | 64 |
| `accum_grad_steps` | Gradient accumulation | 4 |
| `epochs` | Training epochs | 2 |
| `learning_rate` | Initial LR | 2e-4 |
| `warmup_steps` | LR warmup steps | 128 or 0.1 (ratio) |
| `max_grad_norm` | Gradient clipping | 1.0 |
| `stochastic_depth` | Layer dropout prob | 0.1 |

---

## Model Architecture

### Whisper Base Architecture
```
Input Audio (30s @ 16kHz)
    ↓
Mel Spectrogram (80/128 mels × 3000 frames)
    ↓
┌─────────────────────────────────────┐
│         AUDIO ENCODER               │
│  - Conv layers (2x)                 │
│  - Positional embedding             │
│  - Transformer blocks (32 for large)│
│  - Layer normalization              │
└─────────────────────────────────────┘
    ↓ (audio features)
    ├─────────────────────────────────┐
    ↓                                 ↓
┌─────────────────────────────────────┐
│         TEXT DECODER                │
│  - Token embedding                  │
│  - Positional embedding             │
│  - Transformer blocks (32 for large)│
│    (with cross-attention to encoder)│
│  - Layer normalization              │
│  - Output projection (to vocab)     │
└─────────────────────────────────────┘
    ↓
Output Tokens (text + timestamps)
```

### Custom Modifications

#### 1. CheckpointedStochasticAudioEncoder
- Extends `whisper.model.AudioEncoder`
- Adds stochastic depth to each transformer block
- Uses gradient checkpointing to save memory
- Applied to encoder when `gradient_checkpointing_encoder: True`

#### 2. CheckpointedStochasticTextDecoder
- Extends `whisper.model.TextDecoder`
- Adds stochastic depth to each transformer block
- Uses gradient checkpointing to save memory
- Applied to decoder when `gradient_checkpointing_decoder: True`

#### 3. Deep SpecAugment Hooks
- Registered on encoder layer normalizations
- Applies time/frequency masking between layers
- Skips the final layer to allow recovery
- More effective than input-only augmentation

---

## Configuration System

### YAML Configuration Structure

```yaml
model:
  init_name: large-v3-turbo          # Model to fine-tune
  bfloat16: false                    # Use bfloat16 weights
  lora: true                         # Enable LoRA fine-tuning
  lora_config:
    rank: 16                         # LoRA rank
    lora_alpha: 32                   # LoRA scaling factor
    lora_dropout: 0.1                # LoRA dropout

dataset:
  train_datasets: [...]              # HF dataset names
  select_n_per_t_ds: [...]          # Samples per dataset
  groupby_col: [...]                 # Groupby sampling column
  val_datasets: [...]                # Validation datasets
  batch_size: 64
  batch_size_eval: 64
  no_timestamp_training: false       # Disable timestamps
  prompt_use_rate: 0.5               # How often to use prompts
  no_timestamp_rate: 0.5             # Random no-timestamp rate

lr_scheduler:
  type: linear                       # linear, cosine, cosine_with_restarts, etc.
  warmup_steps: 128                  # Warmup steps (or ratio if < 1)

optimizer:
  type: adamw                        # adam or adamw
  8bit: true                         # Use 8-bit optimizer
  params:
    lr: 2.0e-4
    weight_decay: 0.1
    betas: [0.9, 0.98]

training:
  accum_grad_steps: 4                # Gradient accumulation
  epochs: 2
  eval_steps: 0.25                   # Validate every 25% of epoch
  max_grad_norm: 1.0
  stochastic_depth: 0.1
  mixed_precision_training: true
  mp_dtype: fp16                     # fp16 or bfloat16
  gradient_checkpointing_encoder: true
  gradient_checkpointing_decoder: true
  train_only_decoder: false          # Freeze encoder (combine with lora for decoder-only LoRA)
  train_only_encoder: false          # Freeze decoder (combine with lora for encoder-only LoRA)
  save_all_checkpoints: false        # Save intermediate checkpoints
  max_train_loss: 25                 # Abort if loss exceeds this

augmentation:
  spec_augment:
    apply: true
    time_mask_param: 100
    freq_mask_param: 43
    time_warp_w: 80
  deep_spec_augment:
    apply: true
    time_mask_param: 100
    freq_mask_param: 27
    layer_indices: null              # null = all layers except last
  extremes_spec_augment:
    apply: false
    low_freq_range: 10
    high_freq_range: 20
  audio_augment:
    apply_office_aug: true
    apply_baseline_aug: true
    apply_advanced_aug: false        # Filters, pitch shifts, gain changes
    time_stretch:                    # TimeStretch augmentation
      min_rate: 0.75                 # 25% slower (leaves_length_unchanged=False by default)
      max_rate: 1.25                 # 25% faster
  bpe_dropout: 0.1

seed: 123
save_dir: output
```

---

## How to Use

### Installation

```bash
# 1. Clone repository
git clone https://github.com/i4ds/whisper-finetune.git
cd whisper-finetune

# 2. Create virtual environment (Python 3.9+ recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install package
pip install -e .
```

### Training

#### Local Training
```bash
# Single GPU training
python src/whisper_finetune/scripts/finetune.py \
    --config configs/DEBUG.yaml
```

#### SLURM Cluster Training
```bash
# Submit single job
sbatch sc_sbatch.sh configs/large-v3-sg-corpus-mc-1.yaml

# Submit multiple jobs
./multi_submit.sh
```

### Monitoring
Training metrics are logged to Weights & Biases:
- Loss curves (train/validation)
- Word Error Rate (WER)
- Learning rate schedule
- GPU memory usage

### Model Outputs
Models are saved to `{SLURM_JOB_ID}_{save_dir}/` or `{timestamp}_{save_dir}/`:
- `best_model.pt` - Model with lowest validation WER
- `last_model.pt` - Final model after training
- `model.log` - Training log file
- `step{N}.pt` - Intermediate checkpoints (if `save_all_checkpoints: true`)

---

## Deployment

### Model Conversion Pipeline

```
OpenAI Whisper Format (.pt)
    ↓
    convert_openai_to_hf.py
    ↓
HuggingFace Transformers Format
    ↓
    convert_c2t.py
    ↓
CTranslate2 Format (faster-whisper)
```

### Convert to faster-whisper

```bash
python src/whisper_finetune/scripts/convert_c2t.py \
    --model path/to/best_model.pt \
    --out_dir converted_model \
    --quantization float16 \
    --tokenizer_json_path whisper_v3_turbo_utils/tokenizer.json \
    --config_json_path whisper_v3_turbo_utils/config.json \
    --vocabulary_json_path whisper_v3_turbo_utils/vocabulary.json
```

### Inference
Use with [faster-whisper](https://github.com/SYSTRAN/faster-whisper):
```python
from faster_whisper import WhisperModel

model = WhisperModel("converted_model/ct2", device="cuda")
segments, info = model.transcribe("audio.wav")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### Advanced: WhisperX Integration
For improved quality, serve with [WhisperX](https://github.com/m-bain/whisperX):
- Better VAD (Voice Activity Detection)
- Word-level timestamps
- Speaker diarization

---

## Technical Details

### Memory Optimization Techniques

| Technique | Memory Saved | Speed Impact |
|-----------|--------------|--------------|
| Gradient Checkpointing | ~40-50% | -20-30% slower |
| Mixed Precision (FP16) | ~40% | +10-20% faster |
| BFloat16 Weights | ~50% | Minimal |
| 8-bit Optimizer | ~60% | Minimal |
| Gradient Accumulation | Enables larger effective batch | -N% slower (N=accum_steps) |

### Compute Requirements

**Recommended Setup (Large V3):**
- GPU: A100 80GB (or 2× A100 40GB)
- CPU: 8+ cores
- RAM: 64GB
- Storage: 500GB (for datasets + checkpoints)

**Minimal Setup (Turbo V3):**
- GPU: RTX 3090 (24GB)
- CPU: 4+ cores
- RAM: 32GB
- Storage: 200GB

### Training Time Estimates
- **Large V3** on Swiss German (2 epochs):
  - ~60-80 hours on A100 80GB
  - Batch size 64, gradient accumulation 4
  
- **Turbo V3** on Swiss German (2 epochs):
  - ~30-40 hours on A100 80GB
  - Batch size 64, gradient accumulation 4

### Learning Rate Schedules

## Code Locations Reference

### Where is...?

| Feature | File | Function/Class |
|---------|------|----------------|
| Main training loop | `scripts/finetune.py` | `main_loop()`, `main()` |
| Model loading | `model/model_utils.py` | `load_model_and_set_heads()` |
| Training step | `model/model_utils.py` | `train_step()` |
| Evaluation (multi-dataset) | `eval/evaluator.py` | `evaluate_multiple_datasets()` |
| Evaluation (single dataset) | `eval/evaluator.py` | `evaluate_single_dataset()` |
| Metrics computation | `eval/metrics.py` | `compute_*()` functions |
| Dataset class | `data/data_loader.py` | `AudioDataset` |
| Data augmentation | `model/augment.py` | `get_audio_augments_*()` |
| SpecAugment | `data/data_loader.py` | `AudioDataset._calculate_mel()` |
| Deep SpecAugment | `model/model_utils.py` | `register_deep_spec_augment_hooks()` |
| Stochastic depth | `model/model_utils.py` | `CheckpointedStochastic*` classes |
| **LoRA utilities** | `model/lora.py` | `apply_lora()`, `disable_all_but_parametrized_grads()` |
| Optimizer setup | `model/optimizer.py` | `get_optimizer()` |
| LR scheduler | `model/scheduler.py` | `get_scheduler()` |
| Text normalization | `eval/utils.py` | `normalize_text()` |
| Config loading | `utils.py` | `read_config()` |
| Data sampling | `data/utils.py` | `process_dataset()` |

---

## References

### Related Projects
- **Data Preparation:** [i4Ds/whisper-prep](https://github.com/i4Ds/whisper-prep)
- **Inference Engine:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Enhanced Inference:** [WhisperX](https://github.com/m-bain/whisperX)
- **Original Whisper:** [OpenAI Whisper](https://github.com/openai/whisper)

### Research Papers
- **Whisper:** [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **SpecAugment:** [SpecAugment: A Simple Data Augmentation Method for ASR](https://arxiv.org/abs/1904.08779)
- **SpecAugment++:** [SpecAugment++: A Hidden Space Data Augmentation Method](https://arxiv.org/abs/2103.16858)
- **Stochastic Depth:** [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)
- **LoRA:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### Acknowledgments
This repository was based on the excellent work by [Jumon](https://github.com/jumon) at [whisper-finetuning](https://github.com/jumon/whisper-finetuning).
