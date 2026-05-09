# Pixels to Predictions — SmolVLM-500M + QLoRA

Final-project submission for **NYU Deep Learning Spring 2026** ("Pixels to Predictions: DL Vision Challenge"). Fine-tunes `HuggingFaceTB/SmolVLM-500M-Instruct` on a ScienceQA-style multimodal multiple-choice dataset using QLoRA, under a strict 5M trainable-parameter cap.

**Final leaderboard score: 0.82293** (Team: CV Lords)

---

## Task

Given an image, a question, and 2–5 candidate answer choices, predict the correct 0-indexed answer. Inputs include optional `hint` and `lecture` text and pedagogical metadata (`subject`, `topic`, `category`, `skill`).

## Approach

QLoRA fine-tuning of SmolVLM-500M, with:

- **4-bit NF4 quantization** of the base model + double quantization, bf16 compute
- **LoRA on all 7 linear projections** in the text decoder (`q/k/v/o/gate/up/down`), `r=9, α=18` → **4.88M trainable parameters** (under the 5M cap)
- **Vision encoder + modality projector frozen** (no parameter budget for them)
- **Metadata-augmented prompts** — `subject | topic | category | skill` line gives the model a free topical prior
- **Choice-order shuffling** during training (essential — training labels are 37% "A", so without shuffling the model learns to always guess A)
- **Single-token answer-letter target** with loss masking — only the answer-letter token contributes to the loss
- **Single-forward-pass MCQ inference** — score answer-letter logits directly, no generation/parsing
- **8-pass choice-permutation TTA** at inference (+0.7 points typical)
- **Train + val merged** for the final fitting (4,157 examples)

## Setup

Designed to run on a single GPU with at least 16 GB VRAM. Tested on:
- Google Colab Pro+ G4 (NVIDIA RTX PRO 6000 Blackwell, 96 GB)
- ~25 min training, ~5 min inference

```bash
pip install transformers==4.57.6 peft==0.18.1 accelerate datasets pillow num2words
pip install -U bitsandbytes   # important: needs >=0.45 for Blackwell sm_120
```

For Blackwell GPUs (sm_120), restart the runtime once after upgrading bitsandbytes — the broken older shared library stays loaded otherwise.

## Data layout

The notebook expects:

```
DATA_DIR/
├── train.csv
├── val.csv
├── test.csv
├── sample_submission.csv
└── images/
    ├── train/*.png
    ├── val/*.png
    └── test/*.png
```

Edit `DATA_DIR` at the top of the imports cell (default: `/content/data`).

## Running

Open `pixels_to_predictions_qlora.ipynb` and run cells top to bottom. Total time end-to-end is about 35 minutes on a Blackwell or H100 GPU.

The notebook saves:
- `lora_best/` — the best LoRA adapter (selected by held-out val accuracy before the train+val merge)
- `submission.csv` — final predictions in Kaggle format

## Key configuration

| Knob | Value |
|---|---|
| LoRA rank, alpha | r=9, α=18 |
| LoRA target modules | q, k, v, o, gate, up, down (text decoder only) |
| Trainable parameters | 4.88M |
| Quantization | 4-bit NF4 + double quantization |
| Image resolution | 384 (longest edge), splitting disabled |
| Per-device batch | 4 |
| Gradient accumulation | 4 |
| Effective batch | 16 |
| Learning rate | 2e-4, cosine schedule, 5% warmup |
| Optimizer | paged_adamw_8bit |
| Epochs | 5 |
| TTA permutations at inference | 8 |
| Random seed | 42 |

## Results

| | Value |
|---|---|
| Validation accuracy (epoch 4, before train+val merge) | 0.8082 |
| Public LB (train + val merge) | 0.81488 |
| **Final (private) LB** | **0.82293** |

## Repository

```
.
├── pixels_to_predictions_qlora.ipynb   # main notebook (training + inference)
├── README.md                           # this file
└── report/
    └── final_report.tex                # writeup
```

## Authors

- **Udit Gavasane** — `umg215@nyu.edu`
- **Alex Miller** — `amm10358@nyu.edu`

NYU Tandon School of Engineering, CS-GY 9223 / ECE-GY 7123.

## Acknowledgments

Anthropic's Claude was used as a coding assistant for pipeline scaffolding and debugging. Training was run on Google Colab Pro+ with G4 (Blackwell) GPU access.
