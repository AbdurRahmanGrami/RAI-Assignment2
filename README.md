# RAI Assignment 2
## Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety
### FAST-NUCES — Responsible & Explainable AI

---

## Environment

| Item | Value |
|------|-------|
| Python | 3.10 (Google Colab default) |
| GPU | NVIDIA T4 (Google Colab free tier) |
| CUDA | 12.2 |
| Platform | Google Colab |

---

## Project structure
RAI_Assignment2/
├── part1.ipynb                    # Baseline DistilBERT classifier
├── part2.ipynb                    # Bias audit (Black vs. White cohort)
├── part3.ipynb                    # Adversarial attacks
├── part4.ipynb                    # Bias mitigation techniques
├── part5.ipynb                    # Guardrail pipeline demonstration
├── pipeline.py                    # ModerationPipeline class
├── requirements.txt               # Pinned dependencies
├── README.md                      # This file
│
├── train_df.csv                   # 100k stratified training subset
├── eval_df.csv                    # 20k stratified evaluation subset
├── eval_probs_part1.npy           # Part 1 model probabilities
├── eval_probs_best_mitigated.npy  # Best mitigated model probabilities
├── mask_high_black.npy            # High-black cohort boolean mask
├── mask_reference.npy             # Reference cohort boolean mask
│
├── saved_model_part1/             # Fine-tuned DistilBERT (Part 1)
├── saved_model_reweighed/         # Reweighing mitigated model
├── saved_model_oversampled/       # Oversampling mitigated model
├── saved_model_best_mitigated/    # Best model (used in pipeline)
│
├── part4_outputs/                 # All Part 4 artefacts
│   ├── part4_comparison_table.csv
│   ├── part4_pareto_frontier.png
│   ├── part4_comparison_chart.png
│   └── ...
│
└── part5_outputs/                 # All Part 5 artefacts
├── part5_layer_distribution.png
├── part5_threshold_sensitivity.png
├── part5_threshold_sensitivity.csv
└── calibrated_probs.npy

> **Note:** Dataset CSV files, model checkpoints (*.safetensors, *.bin, *.pt),
> and numpy arrays are excluded from the GitHub repository via .gitignore.
> See the Dataset section below for download instructions.

---

## Dataset

**Jigsaw Unintended Bias in Toxicity Classification**
- Source: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
- Files used: `jigsaw-unintended-bias-train.csv` only
- Requires a free Kaggle account and competition acceptance

Download steps:
1. Create a free account at kaggle.com
2. Go to the competition page above
3. Accept the competition rules on the Data tab
4. Download `jigsaw-unintended-bias-train.csv`
5. Place it in your `RAI_Assignment2/` Google Drive folder before running Part 1

---

## How to reproduce

All five parts run on **Google Colab free tier** with a T4 GPU.
Each notebook is self-contained and loads its inputs from Google Drive.

### Step 1 — Prepare Drive
Create a folder called `RAI_Assignment2` in your Google Drive root.
Place `jigsaw-unintended-bias-train.csv` inside it.

### Step 2 — Run notebooks in order

| Notebook | Inputs needed | Approx. runtime |
|----------|--------------|-----------------|
| part1.ipynb | jigsaw-unintended-bias-train.csv | 35 min (training) |
| part2.ipynb | train_df.csv, eval_df.csv, eval_probs_part1.npy | 5 min |
| part3.ipynb | train_df.csv, eval_df.csv, eval_probs_part1.npy, saved_model_part1/ | 35 min (training) |
| part4.ipynb | train_df.csv, eval_df.csv, eval_probs_part1.npy, mask_*.npy | 70 min (2 training runs) |
| part5.ipynb | eval_df.csv, eval_probs_best_mitigated.npy, saved_model_best_mitigated/ | 10 min |

For each notebook:
1. Open in Colab
2. Runtime → Change runtime type → T4 GPU
3. Run Cell 1 (installs), then Runtime → Restart session
4. Run all remaining cells in order

### Step 3 — pipeline.py
`pipeline.py` is written to Drive automatically by Part 5, Cell 6.
To use it standalone:

```python
from pipeline import ModerationPipeline

pipeline = ModerationPipeline(model_path='./saved_model_best_mitigated')
pipeline.fit_calibrator(cal_texts, cal_labels)

result = pipeline.predict("some comment text")
print(result)
# {'decision': 'block', 'layer': 'model', 'confidence': 0.8234}
```

---

## Key results summary

| Part | Key finding |
|------|------------|
| Part 1 | DistilBERT achieves AUC-ROC > 0.95. Optimal threshold = 0.4 due to class imbalance (~8% toxic) |
| Part 2 | High-black cohort FPR is ~2x reference FPR. Disparate Impact ratio > 1.0 confirms over-flagging |
| Part 3 | Character evasion ASR depends on model confidence distribution. Label-flip poisoning raises FNR most |
| Part 4 | All three techniques reduce High-black FPR. Demographic parity and equalized odds are mathematically incompatible when base rates differ |
| Part 5 | Default 0.4–0.6 uncertainty band balances review queue volume against auto-action accuracy |
