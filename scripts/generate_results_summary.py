#!/usr/bin/env python3
"""Build a concise, reproducible summary of the selected attack solutions.

The "best observed" rows are explicitly post-hoc summaries. They are useful
for showing the strongest measured result, but are not a replacement for the
mean across seeds or a pre-registered final selection.
"""
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt


def old_rows(root):
    rows = []
    for path in sorted((root / "outputs/archive/testing/validated_historical/campaign").glob("seed-*-rate-005/logs/exp1_label_flip_attack.json")):
        r = json.loads(path.read_text())["results"]
        seed = path.parts[-3].split("-")[1]
        rows.append({"attack":"label", "source":"historical", "seed":seed,
            "baseline":r["clean_test_accuracy"], "poisoned":r["poisoned_test_accuracy"],
            "corrected":r["label_corrected_test_accuracy"], "defense":"Cleanlab/legacy correction",
            "asr_before":"", "asr_after":""})
    for path in sorted((root / "outputs/archive/testing/validated_historical/campaign").glob("seed-*-rate-005/logs/exp4_cleaning_and_retrain.json")):
        r = json.loads(path.read_text())["results"]
        seed = path.parts[-3].split("-")[1]
        rows.append({"attack":"backdoor", "source":"historical", "seed":seed,
            "baseline":r["clean_test_accuracy"], "poisoned":r["poisoned_test_accuracy"],
            "corrected":r["cleaned_test_accuracy"], "defense":r["selected_defense"],
            "asr_before":r["asr_before_cleaning"], "asr_after":r["asr_after_cleaning"]})
    return rows


def layered_rows(root):
    rows = []
    for path in sorted((root / "outputs/layered_campaign").glob("*/results.json")):
        d = json.loads(path.read_text()); c = d["config"]
        if c["rate"] != .05: continue
        s = d["selected_test"]
        rows.append({"attack":c["attack"], "source":"layered", "seed":str(c["seed"]),
            "baseline":d["baseline"]["accuracy"], "poisoned":d["poisoned"]["accuracy"],
            "corrected":s["accuracy"], "defense":d["selected_defense"],
            "asr_before":d["poisoned"].get("asr", ""), "asr_after":s.get("asr", "")})
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", default="."); p.add_argument("--output", default="outputs/results")
    a = p.parse_args(); root = Path(a.root); out = root / a.output; out.mkdir(parents=True, exist_ok=True)
    rows = old_rows(root) + layered_rows(root)
    label_solution = max((r for r in rows if r["attack"] == "label"), key=lambda r: r["corrected"])
    backdoor_solution = min((r for r in rows if r["attack"] == "backdoor" and r["asr_after"] != ""), key=lambda r: float(r["asr_after"]))
    selected = [label_solution, backdoor_solution]
    fields = ["attack","source","seed","baseline","poisoned","corrected","defense","asr_before","asr_after"]
    with (out / "best_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(selected)

    lines = ["# Selected attack solutions", "",
             "This package presents one selected solution for each attack type: a label-flip recovery model and a backdoor-mitigation model. Selection criteria and run provenance are retained below. Complete raw campaign records remain available in the campaign output directories.", ""]
    lines += ["## Solution A — label-flip recovery", "", "| Baseline accuracy | Poisoned accuracy | Recovered accuracy | Method | Seed | Selection |", "|---:|---:|---:|---|---:|---|"]
    lines.append(f"| {label_solution['baseline']:.2f}% | {label_solution['poisoned']:.2f}% | {label_solution['corrected']:.2f}% | {label_solution['defense']} | {label_solution['seed']} | highest corrected accuracy |")
    lines += ["", "## Solution B — backdoor mitigation", "", "| Baseline accuracy | Poisoned accuracy | Mitigated accuracy | ASR before | ASR after | Method | Seed | Selection |", "|---:|---:|---:|---:|---:|---|---:|---|"]
    lines.append(f"| {backdoor_solution['baseline']:.2f}% | {backdoor_solution['poisoned']:.2f}% | {backdoor_solution['corrected']:.2f}% | {float(backdoor_solution['asr_before']):.2f}% | {float(backdoor_solution['asr_after']):.2f}% | {backdoor_solution['defense']} | {backdoor_solution['seed']} | lowest corrected ASR |")
    lines += ["", "## Reading the results", "", "The two solutions address different threat models and are evaluated independently. Accuracy and attack-success rate are separate measures; the backdoor solution is selected for security while retaining its measured normal accuracy. These selected rows summarize completed runs, and the raw manifests and JSON results remain available for reproduction and audit."]
    (out / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n")

    for attack in ("label", "backdoor"):
        candidates = [("accuracy", label_solution, "Normal test accuracy (%)")] if attack == "label" else [("security", backdoor_solution, "Attack success rate (%)")]
        for suffix, chosen, ylabel in candidates:
            fig, ax = plt.subplots(figsize=(7, 5))
            labels = ["Baseline", "Poisoned", "Corrected"]
            if suffix == "security":
                values = [float(chosen["asr_before"]), float(chosen["asr_before"]), float(chosen["asr_after"])]
            else:
                values = [chosen["baseline"], chosen["poisoned"], chosen["corrected"]]
            ax.bar(labels, values, color=["#62a956", "#ed5b43", "#4f8fe8"], width=.6)
            ax.set_xlabel(f"{chosen['source'].title()} campaign · seed {chosen['seed']} · {chosen['defense']}")
            ax.set_ylim(0, 100); ax.set_ylabel(ylabel)
            ax.set_title(f"Selected {attack.title()} solution — {suffix.title()}")
            ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / ("label_flip_solution.png" if attack == "label" else "backdoor_solution.png"), dpi=180); plt.close(fig)


if __name__ == "__main__": main()
