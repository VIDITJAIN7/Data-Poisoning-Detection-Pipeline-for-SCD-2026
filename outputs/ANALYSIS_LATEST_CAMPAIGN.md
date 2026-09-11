# Latest campaign audit and snapshot

## Completion and evidence

Audited `outputs/archive/testing/validated_historical/campaign`, containing five completed manifests and
25 experiment logs (experiments 0–4 for each condition). All 40 saved training
histories contain 30 epochs. The three internal Cleanlab fold histories are
not saved; the source runs them with the same epoch budget, but their individual
completion cannot be independently audited from those histories.
The terminal log ends with campaign completion and graph generation for seed 42
at 5%. Other conditions contain in-experiment plots, not full comparison graphs.

This snapshot preserves measured outcomes, including failures. It is not a
certification that the previously proposed validated pipeline was implemented.

## Measured normal test accuracy (%)

Use the baseline embedded in the corresponding attack experiment for paired
comparisons. Experiment 0 trains a separate baseline and differs from these.

| Poison rate | Seed | Label baseline | Label poisoned | Label corrected | Backdoor baseline | Backdoor poisoned | Backdoor cleaned | ASR before → after (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 5% | 42 | 89.55 | 84.54 | 88.34 | 89.55 | 89.99 | 89.52 | 95.33 → 1.22 |
| 5% | 43 | 89.22 | 84.04 | 87.51 | 89.21 | 90.95 | 90.11 | 96.04 → 1.28 |
| 5% | 44 | 89.86 | 86.60 | 87.28 | 89.86 | 90.88 | 86.81 | 95.88 → 1.69 |
| 1% | 42 | 89.71 | 90.91 | 90.36 | 89.71 | 89.75 | 88.52 | 65.72 → 1.50 |
| 10% | 42 | 88.59 | 81.35 | 80.54 | 88.58 | 87.96 | 90.08 | 96.09 → 1.27 |

At 5%, mean ± sample standard deviation across three seeds:

| Measure | Result |
|---|---:|
| Label baseline | 89.54 ± 0.32 |
| Label poisoned | 85.06 ± 1.36 |
| Label corrected | 87.71 ± 0.56 |
| Backdoor baseline | 89.54 ± 0.33 |
| Backdoor poisoned | 90.61 ± 0.54 |
| Backdoor cleaned | 88.81 ± 1.76 |
| Backdoor ASR before | 95.75 ± 0.37 |
| Backdoor ASR after | 1.40 ± 0.25 |

Label correction improves 5% accuracy by 2.65 percentage points on average,
but remains 1.83 points below its baseline. Every 5% seed misses the within-one-
point target (gaps 1.21, 1.71, 2.58). Backdoor clean accuracy falls in all three
5% seeds; seed 44 is 3.05 points below baseline. None of the five conditions
meets BOTH requested accuracy criteria for BOTH attacks. Interpret “within 1%”
as an absolute difference of at most one percentage point.

## Detection and attack-specific interpretation

Label correction uses Cleanlab three-fold OOF predictions, a 0.50 confidence
cutoff, and knowledge of the airplane→truck transition. It is not the earlier
clean-reference teacher. At 5%, precision is 99.49%, 98.93%, 99.10%, but recall
is only 31.36%, 25.96%, 31.00%. It changes 788, 656, 782 labels, including 4, 7,
7 false corrections, and leaves 1,716, 1,851, 1,725 poisoned labels unchanged.
Thus precision improved substantially, while recovery is limited by missed
poison. These observations do not establish that partial correction inherently
harms learning; earlier statements attributing large drops to that cause were
not established by controlled experiments.

At 1%, label precision/recall are 87.74%/73.00%, with 416 changes, 365 true
repairs and 51 false repairs. Accuracy falls 0.55 points versus the poisoned
model, though it is 0.65 above baseline. At 10%, all 5,000 airplane labels are
flipped: airplane test accuracy is zero, no corrections are made, and the
0.81-point change comes from retraining unchanged poisoned data. This is a
class-erasure failure, not a successful correction.

At 5%, airplane accuracy drops from 90.7/91.4/92.7 to 44.2/47.8/51.7 for
seeds 42/43/44. Corrected per-class accuracy is not logged, so recovery of the
attacked class cannot be confirmed separately.

Backdoor cleaning selects `known_trigger_signature` in all five runs. At 5%,
it detects all 2,500 inserted patches but also removes 309/306/302 clean images.
Precision is 89.00–89.22%, recall 100%, and FPR approximately 0.64–0.65%.
At 1% precision drops to 61.20% (317 false positives); at 10% it is 94.43%
(295 false positives). Naturally white corners explain why exact white-patch
matching is not perfectly specific. Strong ASR reduction supports this KNOWN
TRIGGER filtering demonstration only; it does not establish unknown-trigger
detection or Neural Cleanse/STRIP performance.

Generic experiment-3 detectors remain weak at 5%:
spectral precision 15.61–19.46%, recall 27.92–36.08%; activation clustering
precision approximately 11%, recall 94.20–96.76%, FPR 40.27–41.04%; loss
precision 4.24–5.16%; KNN precision 4.84–7.04%. These use a separately trained
model from experiment 4, so do not conflate their detector metrics.

## Unresolved methodological and reporting issues

1. Neural Cleanse, STRIP, model unlearning/pruning and ART defense/attack
   integration were promised but are absent from these executed paths.
2. Experiment 4 selects a defense using true poison labels from the same final
   run (F1/FPR). This is oracle-assisted selection, not frozen development-set
   selection. No independent seeds 100/101 calibration exists here.
3. The promised 45k/5k split and matched-count random-removal controls are
   absent. Runs use 50,000 training images and 10,000 test images. Consequently
   superiority over random removal has not been measured.
4. Experiment 1 overwrites the baseline checkpoint; experiment 0's baseline
   logs remain different. Experiments 2/3/4 retrain separate backdoor models.
   Training seeds are not reset for each paired model. Performance differences
   therefore include uncontrolled initialization/training variation.
5. Existing label graphs incorrectly say KNN/model prediction; backdoor graphs
   incorrectly say Spectral union KNN. Differential graphs use experiment 0's
   different baseline. Preserve these as historical artifacts, not selected
   summaries. The tables in this audit use the actual methods and paired values.
6. No triggered true-label accuracy, corrected confusion matrices, baseline
   triggered evaluation or statistical uncertainty beyond three run values is
   available in these logs. ASR alone is not triggered classification accuracy.
7. Subsetting renumbers training IDs; it does preserve missed poisoned
   observations in the current cleaning path, but does not meet stable-ID API
   requirements. Backdoor spatial augmentation still follows patch insertion.
8. Manifests omit dataset/split hashes, dependency versions and intermediate
   checkpoints. OOF predictions and correction ID lists are not persisted.
   First campaign manifest references a5f323c; later ones reference 435a370.
   Their Git diff changes only the campaign graph destination, not training.

## Snapshot contents and next priorities

Preserve current source, this audit, all campaign JSON/PNG artifacts and the
campaign terminal log. Dataset, virtual environment and model weights remain
excluded from Git; model weights remain locally under the campaign directory.
The snapshot is local unless a push is explicitly confirmed successful.

Before further performance tuning: isolate development selection from final
evaluation, implement the promised established backdoor defenses, reset paired
training seeds, reuse baseline/poisoned checkpoints, persist correction evidence,
and add integrity regression tests and random-removal controls. Then regenerate
accurately labeled multi-seed graphs. Do not tune repeatedly on test accuracy to
force the one-point target or present this snapshot as having met that target.
