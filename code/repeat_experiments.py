from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, choices=[
        "best_accuracy",
        "best_accuracy_channel_temporal",
        "best_accuracy_temporal",
        "best_accuracy_eca",
        "best_accuracy_se",
    ])

    parser.add_argument(
        "--protocol",
        required=True,
        choices=["loso", "aos"],
    )

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--channel-reduction-ratio",
        type=int,
        default=16,
        help="Channel-attention reduction ratio r. Used only with best_accuracy_channel_temporal.",
    )
    parser.add_argument(
        "--se-reduction-ratio",
        type=int,
        default=16,
        help="SE reduction ratio r. Used only with best_accuracy_se.",
    )

    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    run_results = []

    for run_index in range(args.runs):

        run_number = run_index + 1
        seed = args.base_seed + run_index

        run_dir = root / f"run_{run_number:02d}"

        print("\n" + "=" * 70)
        print(f"RUN {run_number}/{args.runs}")
        print(f"Model:    {args.model}")
        print(f"Protocol: {args.protocol}")
        print(f"Seed:     {seed}")
        print("=" * 70)

        command = [
            sys.executable,
            "experiments.py",

            "--model",
            args.model,

            "--data-root",
            args.data_root,

            "--protocol",
            args.protocol,

            "--epochs",
            str(args.epochs),

            "--batch-size",
            str(args.batch_size),

            "--learning-rate",
            str(args.learning_rate),

            "--seed",
            str(seed),

            "--channel-reduction-ratio",
            str(args.channel_reduction_ratio),

            "--se-reduction-ratio",
            str(args.se_reduction_ratio),

            "--output-dir",
            str(run_dir),
        ]

        subprocess.run(command, check=True)

        summary_path = (
            run_dir
            / args.protocol
            / "summary.json"
        )

        with open(summary_path, "r") as f:
            summary = json.load(f)

        run_accuracy = summary["mean_test_accuracy"]
        parameter_counts = {
            int(fold["num_trainable_parameters"])
            for fold in summary["folds"]
        }
        if len(parameter_counts) != 1:
            raise RuntimeError(
                f"Expected one parameter count per run, found {sorted(parameter_counts)}"
            )
        num_parameters = parameter_counts.pop()

        run_results.append({
            "run": run_number,
            "seed": seed,
            "mean_test_accuracy": run_accuracy,
            "num_trainable_parameters": num_parameters,
        })

        print(
            f"Run {run_number:02d} finished: "
            f"{run_accuracy * 100:.2f}%"
        )

    accuracies = np.array(
        [r["mean_test_accuracy"] for r in run_results],
        dtype=float,
    )

    final_summary = {
        "model": args.model,
        "protocol": args.protocol,
        "num_runs": args.runs,
        "base_seed": args.base_seed,
        "channel_reduction_ratio": args.channel_reduction_ratio,
        "se_reduction_ratio": args.se_reduction_ratio,
        "seeds": [
            args.base_seed + i
            for i in range(args.runs)
        ],
        "mean_accuracy_across_runs": float(accuracies.mean()),
        "std_accuracy_across_runs": float(accuracies.std()),
        "minimum_run_accuracy": float(accuracies.min()),
        "maximum_run_accuracy": float(accuracies.max()),
        "runs": run_results,
    }

    output_file = root / "20_run_summary.json"

    with open(output_file, "w") as f:
        json.dump(final_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL 20-RUN RESULT")
    print("=" * 70)

    print(f"Model:    {args.model}")
    print(f"Protocol: {args.protocol}")
    print(f"Runs:     {args.runs}")
    if args.model == "best_accuracy_channel_temporal":
        print(f"Channel r: {args.channel_reduction_ratio}")
    if args.model == "best_accuracy_se":
        print(f"SE r:     {args.se_reduction_ratio}")

    print(
        f"Accuracy: "
        f"{accuracies.mean() * 100:.2f}% "
        f"± {accuracies.std() * 100:.2f}%"
    )

    print(f"Parameters: {num_parameters}")

    print(
        f"Minimum:  {accuracies.min() * 100:.2f}%"
    )

    print(
        f"Maximum:  {accuracies.max() * 100:.2f}%"
    )

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
