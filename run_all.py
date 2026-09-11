"""
Run all experiments sequentially.

Usage:
    python run_all.py              # run everything (exp0–exp5)
    python run_all.py 0 2 3        # run only experiments 0, 2, and 3
"""

import sys
import time
import argparse
import os
import json
import subprocess


EXPERIMENTS = {
    0: ("Clean Baseline", "experiments.exp0_clean_baseline"),
    1: ("Label-Flip Attack", "experiments.exp1_label_flip_attack"),
    2: ("Backdoor Attack", "experiments.exp2_backdoor_attack"),
    3: ("Detection Pipeline", "experiments.exp3_detection_pipeline"),
    4: ("Cleaning & Retrain", "experiments.exp4_cleaning_and_retrain"),
    5: ("Full Demo", "experiments.exp5_full_demo"),
}


def main():
    parser = argparse.ArgumentParser(description="Run poisoning experiments")
    parser.add_argument("experiments", nargs="*", type=int, choices=EXPERIMENTS.keys())
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--poison-rate", type=float, default=None)
    parser.add_argument("--run-dir", default=None,
                        help="directory for this run's logs, plots and models")
    parser.add_argument("--resume", action="store_true",
                        help="reserved for manifest-compatible completed runs")
    args = parser.parse_args()
    import config
    if args.seed is not None:
        config.SEED = args.seed
    if args.poison_rate is not None:
        if not 0.0 <= args.poison_rate <= 1.0:
            parser.error("--poison-rate must be between 0 and 1")
        config.POISON_RATE = args.poison_rate
    if args.run_dir:
        config.OUTPUT_DIR = os.path.abspath(args.run_dir)
        config.LOG_DIR = os.path.join(config.OUTPUT_DIR, "logs")
        config.PLOT_DIR = os.path.join(config.OUTPUT_DIR, "plots")
        config.MODEL_DIR = os.path.join(config.OUTPUT_DIR, "models")
        for path in (config.LOG_DIR, config.PLOT_DIR, config.MODEL_DIR):
            os.makedirs(path, exist_ok=True)
    selected = args.experiments or list(EXPERIMENTS.keys())

    manifest_path = os.path.join(config.OUTPUT_DIR, "run_manifest.json")
    manifest_key = {
        "experiments": selected,
        "seed": config.SEED,
        "poison_rate": config.POISON_RATE,
        "epochs": config.EPOCHS,
    }
    try:
        manifest_key["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest_key["git_revision"] = "unknown"
    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            previous = json.load(f)
        if previous.get("config") != manifest_key:
            parser.error("--resume requested, but the existing manifest configuration differs")
        completed_prior = set(previous.get("completed_experiments", []))
        selected = [i for i in selected if i not in completed_prior]
    elif args.resume and not os.path.exists(manifest_path):
        parser.error("--resume requested, but no run_manifest.json exists")
    else:
        with open(manifest_path, "w") as f:
            json.dump({"status": "running", "config": manifest_key}, f, indent=2)

    print("╔" + "═" * 58 + "╗")
    print("║   DATA POISONING & DETECTION — EXPERIMENT RUNNER         ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Experiments to run: {selected}")
    print(f"  Logs directory: {config.LOG_DIR}\n")

    completed = list(completed_prior) if args.resume else []
    for exp_id in selected:
        if exp_id not in EXPERIMENTS:
            print(f"  [SKIP] Unknown experiment ID: {exp_id}")
            continue

        name, module_path = EXPERIMENTS[exp_id]
        print(f"\n{'#' * 60}")
        print(f"#  Experiment {exp_id}: {name}")
        print(f"{'#' * 60}\n")

        t0 = time.time()
        mod = __import__(module_path, fromlist=["run"])
        mod.run()
        completed.append(exp_id)
        elapsed = time.time() - t0

        print(f"  ⏱  Experiment {exp_id} completed in {elapsed:.1f}s\n")

    with open(manifest_path, "w") as f:
        json.dump({"status": "complete", "config": manifest_key,
                   "completed_experiments": completed}, f, indent=2)
    print("All selected experiments finished.")
    print("Run `python scripts/generate_graphs.py` to produce comparison charts from logs.")


if __name__ == "__main__":
    main()
