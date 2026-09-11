#!/usr/bin/env python3
"""Generate summaries only from complete trusted-campaign runs."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    args = p.parse_args(); root = Path(args.output)
    rows = []
    for path in sorted(root.glob("*-seed-*/results.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "complete": continue
        cfg = data["config"]
        rows.append((cfg, data))
    if not rows: raise SystemExit("No complete results found")
    lines = ["attack,rate,seed,baseline_acc,poisoned_acc,corrected_or_selected_acc,poisoned_asr,corrected_asr,selected_defense"]
    for cfg, data in rows:
        selected = data.get("selected_defense", "")
        selected_metrics = data.get("selected_test", {})
        lines.append(",".join(map(str, [cfg["attack"], cfg["rate"], cfg["seed"],
            round(data["baseline"]["accuracy"], 4), round(data["poisoned"]["accuracy"], 4),
            round(selected_metrics.get("accuracy", float("nan")), 4),
            data.get("poisoned", {}).get("asr", ""), selected_metrics.get("asr", ""), selected])))
    (root / "results_summary.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    for attack in ("label", "backdoor"):
        chosen = [(c, d) for c, d in rows if c["attack"] == attack and c["rate"] == .05]
        if not chosen: continue
        labels = ["Baseline", "Poisoned", "Corrected"]
        values = [[], [], []]
        for c, d in chosen:
            selected = d.get("candidate_scores", {}).get(d.get("selected_defense", ""), {})
            values[0].append(d["baseline"]["accuracy"]); values[1].append(d["poisoned"]["accuracy"])
            values[2].append(selected.get("accuracy", np.nan))
        means = [np.nanmean(x) for x in values]; std = [np.nanstd(x, ddof=1) if len(x) > 1 else 0 for x in values]
        fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(labels, means, yerr=std, capsize=5, color=["#62a956", "#ed5b43", "#4f8fe8"])
        ax.set_ylabel("Test accuracy (%)"); ax.set_title(f"{attack.title()} defense — 5% poisoning")
        ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.25)
        for i, v in enumerate(means): ax.text(i, v + std[i] + 1, f"{v:.2f}%", ha="center")
        fig.tight_layout(); fig.savefig(root / f"{attack}_accuracy_summary.png", dpi=180); plt.close(fig)


if __name__ == "__main__": main()
