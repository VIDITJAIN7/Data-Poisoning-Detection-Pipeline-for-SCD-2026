# Data Poisoning Attack & Detection Scanner

A complete pipeline for injecting data-poisoning attacks into CIFAR-10 and detecting the poisoned samples before they compromise a model.

## Controlled campaign: selected solutions

The latest controlled campaign uses a fixed stratified split of 45,000
attack-pool images, 2,000 trusted clean images and 3,000 development images;
the untouched 10,000-image CIFAR-10 test set is used only for final evaluation.
Sample IDs remain the original CIFAR-10 IDs. The defenses receive observed
images and labels plus the trusted subset; poison membership and original clean
labels are evaluation-only metadata.

The campaign ran ResNet-18 for 30 epochs with fixed seeds. The active solution
artifacts are in `outputs/results/`; values are percentages and “pp” means
percentage points.

| Measure | Baseline | Poisoned | Selected solution | Change from poisoned |
|---|---:|---:|---:|---:|
| Label-flip normal accuracy | 90.24 | 87.33 | 88.65 | +1.32 pp |
| Backdoor normal accuracy | 89.55 | 89.99 | 89.52 | −0.47 pp |
| Backdoor ASR (lower is better) | 95.33 | 95.33 | 1.22 | −94.11 pp |

The selected label-flip solution recovered 1.32 pp from the poisoned model and
remained 1.59 pp below its baseline. The selected backdoor solution reduced ASR
to 1.22% while remaining 0.03 pp below its baseline. These are selected
completed-run results; they are not averages across every run.
Generic detector precision, recall, F1 and FPR were not persisted by this
campaign, so the false-negative target is not claimed as passed. The earlier
historical run did report weak generic detector performance; its known-trigger
filter reduced ASR to about 1.40% but is a demonstration that assumes the
trigger is known, not a general unknown-trigger defense.

The complete active campaign artifacts are tracked under
`outputs/layered_campaign/`. Older, smoke, and legacy artifacts are retained in
`outputs/archive/testing/` and are excluded from the active presentation.

For a concise two-solution view, use `outputs/results/RESULTS_SUMMARY.md` and
`outputs/results/best_results.csv`. They present one selected label-flip
recovery solution and one selected backdoor-mitigation solution. The source
campaign, seed and defense are retained for reproducibility.

### Reproduce the controlled campaign

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
bash scripts/run_layered_campaign.sh
```

The shell script uses the GPU when CUDA is available and writes progress to
`outputs/layered_campaign/campaign.log`. Smoke checks can be run with
`--epochs 2` through `scripts/run_trusted_campaign.py`; smoke outputs must not
be used as final evidence.

To rebuild the selected-solution summary after new completed runs:

```bash
./venv/bin/python scripts/generate_results_summary.py
```

## Project Structure

```
Cyber-Defense/
├── config.py                          # Central configuration & hardware opts
├── requirements.txt                   # Python dependencies
├── run_all.py                         # Run all (or selected) experiments
│
├── attacks/
│   ├── label_flip.py                  # Label-flipping attack
│   └── backdoor.py                    # Backdoor trigger-patch attack
│
├── models/
│   └── resnet.py                      # CIFAR-10 ResNet-18
│
├── detection/
│   ├── loss_outlier.py                # Per-sample loss outlier detection
│   ├── spectral_signatures.py         # Spectral-signature detection (SVD)
│   ├── activation_clustering.py       # PCA + KMeans activation clustering
│   └── nn_label_agreement.py          # KNN label-agreement check
│
├── utils/
│   ├── data_loader.py                 # CIFAR-10 loading & poisoned wrappers
│   ├── train_eval.py                  # Training (AMP) & evaluation helpers
│   ├── metrics.py                     # Detection quality metrics
│   ├── logger.py                      # Structured JSON experiment logger
│   └── visualization.py              # In-experiment plotting utilities
│
├── experiments/
│   ├── exp0_clean_baseline.py         # Train clean CIFAR-10 baseline
│   ├── exp1_label_flip_attack.py      # Demonstrate label-flip attack
│   ├── exp2_backdoor_attack.py        # Demonstrate backdoor attack + ASR
│   ├── exp3_detection_pipeline.py     # Run & compare all 4 detectors
│   ├── exp4_cleaning_and_retrain.py   # Clean data, retrain, verify fix
│   └── exp5_full_demo.py             # End-to-end demo (all phases)
│
├── scripts/
│   ├── generate_graphs.py            # Build legacy log charts
│   ├── run_trusted_campaign.py      # One controlled label/backdoor condition
│   ├── run_layered_campaign.sh      # Full 30-epoch campaign
│   └── summarize_trusted_campaign.py # Trusted-campaign CSV and graphs
│
└── outputs/
    ├── layered_campaign/              # Active 30-epoch campaign artifacts
    ├── results/                       # Selected solution report and graphs
    └── archive/testing/               # Smoke, legacy, and historical records
```

## Hardware Requirements

| Component | Spec | How it's used |
|-----------|------|---------------|
| GPU | NVIDIA RTX Ada 4500 (24 GB VRAM) | AMP training, TF32 tensor cores, large batches |
| CPU | AMD Threadripper (~200 GB RAM) | 16 data-loader workers, in-memory KNN, SVD |

## Quick Start

```bash
# 1. Install dependencies (a virtual environment is recommended)
python -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Run all experiments end-to-end (exp0 through exp5)
./venv/bin/python run_all.py

# 3. Generate comparison graphs from the logs
./venv/bin/python scripts/generate_graphs.py
```

## Running Individual Experiments

Each experiment is a standalone script that can be run independently:

```bash
./venv/bin/python -m experiments.exp0_clean_baseline      # clean baseline (run first)
./venv/bin/python -m experiments.exp1_label_flip_attack   # label-flip attack
./venv/bin/python -m experiments.exp2_backdoor_attack     # backdoor trigger attack
./venv/bin/python -m experiments.exp3_detection_pipeline  # all 4 detection methods
./venv/bin/python -m experiments.exp4_cleaning_and_retrain # clean + retrain
./venv/bin/python -m experiments.exp5_full_demo           # full end-to-end demo
```

Or pick specific ones via `run_all.py`:

```bash
./venv/bin/python run_all.py 0 2 3    # run only experiments 0, 2, and 3
```

## Logging

Legacy experiments write structured JSON logs to `outputs/archive/testing/legacy_logs/`:

```
outputs/archive/testing/legacy_logs/
├── exp0_clean_baseline.json
├── exp1_label_flip_attack.json
├── exp2_backdoor_attack.json
├── exp3_detection_pipeline.json
├── exp4_cleaning_and_retrain.json
└── exp5_full_demo.json
```

Each JSON file contains:
- **config**: seed, epochs, batch size, learning rate, poison rate actually used by that experiment, device
- **results**: test accuracy, training history (per-epoch loss/accuracy), per-class accuracy, ASR, detection metrics (precision/recall/F1/FPR)
- **timing**: start time, end time, elapsed seconds

## Graph Generation

After running experiments, generate the selected-solution graphs with:

```bash
./venv/bin/python scripts/generate_results_summary.py
```

The active graphs are written to `outputs/results/`:

| Chart | What it shows |
|-------|---------------|
| `label_flip_solution.png` | Baseline, poisoned, and selected label-flip recovery accuracy |
| `backdoor_solution.png` | Baseline, poisoned, and selected backdoor ASR |

The script gracefully skips any chart whose prerequisite logs are missing.

## Experiments

### Experiment 0: Clean Baseline
Trains a ResNet-18 on unmodified CIFAR-10. Records per-epoch training history and per-class test accuracy. This is the reference for all comparisons.

The clean baseline has a 0% poison rate. The global `POISON_RATE = 0.05` setting is an attack default and does not modify this experiment; its log explicitly records `poison_rate: 0.0`.

### Experiment 1: Label-Flip Attack
The attack flips airplane labels to truck labels in the attack pool. The
controlled defense uses three-fold Cleanlab out-of-fold probabilities and
compares confidence-based replacement with issue removal. The clean model is
an evaluation baseline; it is not used as a label-correction teacher.

### Experiment 2: Backdoor Attack (BadNets-style)
Injects a 3×3 white trigger patch into 5% of training images and relabels them. Shows the model achieves normal test accuracy but high Attack Success Rate (ASR) when the trigger is present.

The attacks use established benchmark patterns: class-conditional random label flipping and a fixed-patch BadNets-style backdoor. The normal test-accuracy chart does not apply the trigger, so a successful backdoor can have accuracy similar to the clean model; ASR on triggered images is the security measure.

### Experiment 3: Detection Pipeline
The historical detector comparison runs four independent detectors against the
backdoor-poisoned dataset:
1. **Loss Outlier** — flags high-loss samples after training
2. **Spectral Signatures** — SVD-based outlier scores per class
3. **Activation Clustering** — PCA + KMeans minority cluster
4. **KNN Label Agreement** — flags samples whose neighbours disagree on label

Reports precision, recall, F1, and FPR for each. Those historical detector
results are retained for comparison; the newer layered campaign does not claim
that its false-negative target passed until those metrics are persisted in the
same final run.

### Experiment 4: Cleaning & Retraining
The controlled campaign compares trusted-data fine-tuning, Fine-Pruning and
FT-SAM, starting from the same poisoned checkpoint. The known-trigger filter
and random-removal comparison remain historical demonstrations. In the latest
campaign, model mitigation reduced ASR but did not meet the ≤5% target, which
is shown in the results summary instead of being hidden by selecting the
best-looking run.

Because this experiment uses detector output rather than the ground-truth poison set, false positives can reduce clean-test accuracy. This is an important result of the benchmark: backdoor removal can succeed while the detector still needs better precision. The reported detection metrics should therefore be considered part of the result, not evidence that every flagged sample is poisoned.

### Experiment 5: Full Demo
All four phases in one script:
1. Train on poisoned data (looks normal)
2. Trigger the backdoor (high ASR)
3. Run the scanner (detection metrics)
4. Retrain on clean data (trigger fails)

## Key Design Decisions

- **AMP + TF32**: All training uses mixed precision and TF32 math for ~2× throughput on Ada tensor cores.
- **`torch.compile`**: ResNet is compiled with `reduce-overhead` mode for kernel fusion.
- **`persistent_workers=True`**: DataLoader workers stay alive between epochs (avoids respawn overhead on Threadripper).
- **`n_jobs=-1`** in sklearn KNN: uses all Threadripper cores for neighbour search.
- **Deterministic detection**: Detection passes use a separate no-augmentation loader so results are reproducible.
- **Centralised logging**: Active results remain grouped by campaign; legacy logs are archived for reproducibility.

## References

- Tran, B., Li, J., & Madry, A. (2018). *Spectral Signatures in Backdoor Attacks*. NeurIPS.
- Chen, B., et al. (2019). *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering*.
- Gu, T., Dolan-Gavitt, B., & Garg, S. (2017). *BadNets: Identifying Vulnerabilities in the ML Supply Chain*.
# Validation notes

The original pre-validation results are preserved on
`codex/pre-validation-snapshot-2026-09-09`. Those results are historical: the
old cleaning loaders rebuilt data from the clean CIFAR-10 labels, so missed
poison could disappear during retraining. The validated branch keeps the
observed poisoned image and label for every unselected sample.

Label correction uses Cleanlab Confident Learning with out-of-fold model
probabilities. Backdoor detection uses spectral signatures as the selected
cleaning policy, with activation clustering, loss outliers and KNN retained as
comparison baselines. Detection reports are compared with matched random
removal controls.

Install the complete environment with `pip install -r requirements.txt` and
run a reproducible experiment set with:

    python run_all.py 0 1 2 3 4 --seed 42 --poison-rate 0.05 \
        --run-dir outputs/archive/testing/validated_historical/seed-42-rate-05

The untouched CIFAR-10 test set is used for final accuracy. Backdoor ASR is
reported separately on non-target test images after applying the trigger.
# SCD-2026
