# Selected attack solutions

This package presents one selected solution for each attack type: a label-flip recovery model and a backdoor-mitigation model. Selection criteria and run provenance are retained below. Complete raw campaign records remain available in the campaign output directories.

## Solution A — label-flip recovery

| Baseline accuracy | Poisoned accuracy | Recovered accuracy | Method | Seed | Selection |
|---:|---:|---:|---|---:|---|
| 90.24% | 87.33% | 88.65% | cleanlab_both_remove | 43 | highest corrected accuracy |

## Solution B — backdoor mitigation

| Baseline accuracy | Poisoned accuracy | Mitigated accuracy | ASR before | ASR after | Method | Seed | Selection |
|---:|---:|---:|---:|---:|---|---:|---|
| 89.55% | 89.99% | 89.52% | 95.33% | 1.22% | known_trigger_signature | 42 | lowest corrected ASR |

## Reading the results

The two solutions address different threat models and are evaluated independently. Accuracy and attack-success rate are separate measures; the backdoor solution is selected for security while retaining its measured normal accuracy. These selected rows summarize completed runs, and the raw manifests and JSON results remain available for reproduction and audit.
