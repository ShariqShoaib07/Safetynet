"""
run_eval.py

Validates the Safety Planning LLM against handcrafted world_diff.yaml
scenarios BEFORE closing the loop with the live camera -- per your
supervisor's/collaborator's recommendation: test the LLM's response to
diff YAML in isolation first, with a ground truth, checking both
accuracy and run-to-run consistency, and only then wire it to the live
YOLO pipeline.

Two things this measures, per scenario, per model:
  1. ACCURACY   -- does the verdict match ground_truth.yaml? (skipped/
                    noted separately for "flagged"-confidence scenarios,
                    since those don't have a settled correct answer yet)
  2. CONSISTENCY -- run the same scenario N times; do all N runs agree
                    with each other, regardless of order?

Two history modes, both run, so order-dependence is visible rather than
hidden:
  - "isolated" -- fresh DialogueHistory per scenario call (no cross-talk)
  - "sequential" -- one shared DialogueHistory across the whole shuffled
     run, so if earlier scenarios are biasing later verdicts, it'll show
     up as a consistency gap between the two modes for the same scenario

Usage:
    python run_eval.py                  # all models in models.yaml, 3 repeats
    python run_eval.py --repeats 5
    python run_eval.py --models gpt-4o   # just one model, by its `name` in models.yaml

Output: eval/results/<timestamp>_results.csv (one row per model x
scenario x history_mode x repeat) and a printed summary table.
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safety_checker import SafetyChecker
from dialogue_history import DialogueHistory

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_DIR = os.path.join(EVAL_DIR, "scenarios")
GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, "ground_truth.yaml")
MODELS_PATH = os.path.join(EVAL_DIR, "models.yaml")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def load_scenarios() -> dict:
    scenarios = {}
    for fname in sorted(os.listdir(SCENARIOS_DIR)):
        if not fname.endswith(".yaml"):
            continue
        name = fname[:-5]
        with open(os.path.join(SCENARIOS_DIR, fname), "r", encoding="utf-8") as f:
            scenarios[name] = yaml.safe_load(f)
    return scenarios


def load_ground_truth() -> dict:
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(filter_names=None) -> list[dict]:
    with open(MODELS_PATH, "r", encoding="utf-8") as f:
        models = yaml.safe_load(f) or []
    if filter_names:
        models = [m for m in models if m["name"] in filter_names]
    return models


def run(repeats: int, filter_names=None):
    scenarios = load_scenarios()
    ground_truth = load_ground_truth()
    models = load_models(filter_names)

    if not models:
        print("[run_eval] No models matched -- check models.yaml / --models filter.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"{ts}_results.csv")

    rows = []  # for CSV + summary

    for model_cfg in models:
        print(f"\n=== Model: {model_cfg['name']} ({model_cfg['provider']}/{model_cfg['model']}) ===")
        checker = SafetyChecker(model=model_cfg["model"], api_key_env=model_cfg["api_key_env"])
        checker.client.provider = model_cfg["provider"]

        for history_mode in ("isolated", "sequential"):
            shared_history = DialogueHistory(max_turns=10)  # only used in "sequential" mode

            # Build repeats * scenario order, shuffled, so ordering isn't
            # always the same alphabetical run every time.
            run_order = list(scenarios.keys()) * repeats
            random.shuffle(run_order)

            for scenario_name in run_order:
                diff_state = scenarios[scenario_name]
                history = shared_history if history_mode == "sequential" else DialogueHistory(max_turns=10)

                verdict = checker.check_event(diff_state, history)

                gt = ground_truth.get(scenario_name, {})
                expected = gt.get("expected_verdict")
                confidence = gt.get("confidence", "unknown")
                correct = (verdict["verdict"] == expected) if expected else None

                rows.append({
                    "model": model_cfg["name"],
                    "history_mode": history_mode,
                    "scenario": scenario_name,
                    "expected_verdict": expected,
                    "gt_confidence": confidence,
                    "actual_verdict": verdict["verdict"],
                    "correct": correct,
                    "reason": verdict["reason"],
                })

    # --- write CSV ---
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[run_eval] Wrote {len(rows)} rows to {results_path}")

    _print_summary(rows)


def _print_summary(rows: list[dict]):
    # Accuracy: only over rows where expected_verdict is set AND
    # gt_confidence == "high" (flagged scenarios are excluded from the
    # accuracy score on purpose -- see ground_truth.yaml).
    # Consistency: for each (model, history_mode, scenario), what fraction
    # of repeats agree with the MAJORITY verdict for that group.
    print("\n--- ACCURACY (high-confidence ground truth only) ---")
    acc_groups = defaultdict(lambda: [0, 0])  # (model, history_mode) -> [correct, total]
    for r in rows:
        if r["gt_confidence"] != "high" or r["expected_verdict"] is None:
            continue
        key = (r["model"], r["history_mode"])
        acc_groups[key][1] += 1
        if r["correct"]:
            acc_groups[key][0] += 1
    for (model, mode), (correct, total) in sorted(acc_groups.items()):
        pct = 100 * correct / total if total else 0
        print(f"  {model:16s} [{mode:10s}]  {correct}/{total}  ({pct:.0f}%)")

    print("\n--- CONSISTENCY (agreement with majority verdict, per scenario) ---")
    cons_groups = defaultdict(list)  # (model, history_mode, scenario) -> [verdicts]
    for r in rows:
        key = (r["model"], r["history_mode"], r["scenario"])
        cons_groups[key].append(r["actual_verdict"])
    for (model, mode, scenario), verdicts in sorted(cons_groups.items()):
        majority = max(set(verdicts), key=verdicts.count)
        agree = verdicts.count(majority)
        total = len(verdicts)
        flag = "  <-- INCONSISTENT" if agree < total else ""
        print(f"  {model:16s} [{mode:10s}] {scenario:32s} {agree}/{total} agree on '{majority}'{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3,
                        help="How many times each scenario is presented per model/history-mode")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Filter to specific model names from models.yaml (default: all)")
    args = parser.parse_args()
    run(repeats=args.repeats, filter_names=args.models)
