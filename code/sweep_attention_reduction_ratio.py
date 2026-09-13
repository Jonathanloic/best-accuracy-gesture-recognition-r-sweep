from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_R_VALUES = (2, 4, 8, 16)
MODELS = (
    ("best_accuracy_channel_temporal", "channel_temporal"),
    ("best_accuracy_se", "se"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="20-run reduction-ratio sweep for Best Accuracy Channel+Temporal and SE attention."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", required=True, choices=("loso", "aos"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sweep_results = []

    print("=" * 78)
    print("CHANNEL+TEMPORAL / SE REDUCTION-RATIO SWEEP")
    print("=" * 78)
    print(f"Protocol: {args.protocol.upper()}")
    print(f"Models: Channel+Temporal, SE")
    print(f"r values: {list(DEFAULT_R_VALUES)}")
    print(f"Runs per model/r: {args.runs}")
    print(f"Seeds: {args.base_seed}-{args.base_seed + args.runs - 1}")
    print("=" * 78)

    for model_name, model_dir_name in MODELS:
        for r in DEFAULT_R_VALUES:
            r_dir = root / model_dir_name / f"r_{r}"
            ratio_args = (
                ["--channel-reduction-ratio", str(r)]
                if model_name == "best_accuracy_channel_temporal"
                else ["--se-reduction-ratio", str(r)]
            )

            command = [
                sys.executable,
                "repeat_experiments.py",
                "--model", model_name,
                "--protocol", args.protocol,
                "--data-root", args.data_root,
                "--output-dir", str(r_dir),
                "--runs", str(args.runs),
                "--base-seed", str(args.base_seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                *ratio_args,
            ]

            print("\n" + "=" * 78)
            print(f"STARTING {model_dir_name} | r = {r}")
            print("=" * 78)
            subprocess.run(command, check=True)

            summary_path = r_dir / "20_run_summary.json"
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            parameter_counts = {
                int(run["num_trainable_parameters"])
                for run in summary["runs"]
            }
            if len(parameter_counts) != 1:
                raise RuntimeError(
                    f"Expected one parameter count for {model_name}, r={r}; "
                    f"found {sorted(parameter_counts)}"
                )

            sweep_results.append(
                {
                    "model": model_name,
                    "attention": model_dir_name,
                    "r": r,
                    "num_trainable_parameters": parameter_counts.pop(),
                    "mean_accuracy_across_runs": summary["mean_accuracy_across_runs"],
                    "std_accuracy_across_runs": summary["std_accuracy_across_runs"],
                    "minimum_run_accuracy": summary["minimum_run_accuracy"],
                    "maximum_run_accuracy": summary["maximum_run_accuracy"],
                    "seeds": summary["seeds"],
                }
            )

    final_summary = {
        "models": [name for name, _ in MODELS],
        "protocol": args.protocol,
        "r_values": list(DEFAULT_R_VALUES),
        "runs_per_model_and_r": args.runs,
        "base_seed": args.base_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "results": sweep_results,
    }

    output_file = root / "attention_reduction_ratio_sweep_summary.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)

    print("\n" + "=" * 78)
    print("FINAL REDUCTION-RATIO COMPARISON")
    print("=" * 78)
    print(f"{'Model':<20} | {'r':>3} | {'Params':>8} | {'Mean':>10} | {'Std':>8}")
    print("-" * 78)
    for result in sweep_results:
        label = "Channel+Temporal" if result["attention"] == "channel_temporal" else "SE"
        print(
            f"{label:<20} | {result['r']:>3} | "
            f"{result['num_trainable_parameters']:>8} | "
            f"{result['mean_accuracy_across_runs'] * 100:>9.2f}% | "
            f"{result['std_accuracy_across_runs'] * 100:>7.2f}%"
        )
    print("-" * 78)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
