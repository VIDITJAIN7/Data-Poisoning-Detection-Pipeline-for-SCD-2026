# Layered campaign results

This report records the complete campaign in `outputs/layered_campaign`.
Every listed condition has a complete manifest and 30-epoch training histories.
The test set was not used to choose a defense; candidate selection used the
development split within each condition.

## Final 5% results

| Attack | Seed | Baseline accuracy | Poisoned accuracy | Selected correction | Corrected accuracy | ASR after |
|---|---:|---:|---:|---|---:|---:|
| Label flip | 42 | 89.35% | 86.95% | Cleanlab both remove | 87.60% | — |
| Label flip | 43 | 90.24% | 87.33% | Cleanlab both remove | 88.65% | — |
| Label flip | 44 | 87.24% | 85.56% | Cleanlab prune remove | 87.10% | — |
| Backdoor | 42 | 89.35% | 89.99% | FT-SAM ρ=0.10 | 89.24% | 63.84% |
| Backdoor | 43 | 90.24% | 90.47% | Fine-Pruning 20% | 89.49% | 93.00% |
| Backdoor | 44 | 87.24% | 89.20% | FT-SAM ρ=0.05 | 88.12% | 94.01% |

At 5%, label accuracy averaged 88.94% baseline, 86.61% poisoned and 87.78%
corrected. Backdoor accuracy averaged 88.94%, 89.89% and 88.95%; ASR averaged
96.31% before and 83.62% after mitigation. Thus label recovery improved
accuracy by 1.17 pp but remained 1.16 pp below baseline. Backdoor normal
accuracy met the one-point average target, while ASR did not meet the ≤5%
target.

## Stress checks

At 1% poisoning, label accuracy was 89.78% poisoned and 89.08% corrected
against an 89.35% baseline. Backdoor ASR changed from 91.21% to 86.48%.
At 10%, label accuracy was 78.88% poisoned and 79.84% corrected against an
89.35% baseline. Backdoor ASR changed from 96.99% to 13.21%.

## Limitations

The campaign records Cleanlab issue counts and candidate choices, but the final
results JSON does not persist poison masks or TP/FP/FN detector accounting.
Therefore precision, recall, F1 and false-negative targets cannot be claimed as
passed. Development candidates were selected independently per condition;
development seeds 100 and 101 were not used to freeze one configuration before
the final seeds. FT-SAM and Fine-Pruning are model mitigation comparisons, not
proof of unknown-trigger detection. Historical known-trigger results are kept
separate because they use a stronger assumption.

Use `results_summary.csv` for machine-readable values and the two PNG files for
the accuracy figures. Do not treat a high normal accuracy as
evidence that a backdoor is absent; ASR is the relevant triggered metric.
