from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from dataset import (
    CALIBRATION_SESSION,
    dataset_summary,
    make_aos_split,
    make_loso_split,
    scan_dataset,
    subjects,
    validate_sample_arrays,
)
from train import TrainingConfig, run_fold


BEST_ACCURACY_CONFIG = {
    "name": "best_accuracy",
    "preprocessed_data": {
        "band_pass_hz": [225, 375],
        "downsampling_ratio": 1,
        "window_ms": 1250,
        "normalization": "already present in saved samples; do not reapply",
    },
    "model": {
        "kernel_size": 15,
        "num_blocks": 6,
        "channel_width": 32,
        "widths": (32, 32, 32, 32, 32, 32),
        "dropout": 0.2,
        "input_channels": 4,
        "num_classes": 6,
    },
}

BEST_ACCURACY_CHANNEL_TEMPORAL_CONFIG = {
    "name": "best_accuracy_channel_temporal",
    "base_model": "best_accuracy",
    "preprocessed_data": BEST_ACCURACY_CONFIG["preprocessed_data"],
    "model": {
        **BEST_ACCURACY_CONFIG["model"],
        "attention": {
            "order": "channel_then_temporal",
            "placement": "after_relu_before_dropout",
            "channel_reduction_ratio": 16,
            "temporal_kernel_size": 7,
        },
    },
}



BEST_ACCURACY_SE_CONFIG = {
    "name": "best_accuracy_se",
    "base_model": "best_accuracy",
    "preprocessed_data": BEST_ACCURACY_CONFIG["preprocessed_data"],
    "model": {
        **BEST_ACCURACY_CONFIG["model"],
        "attention": {
            "type": "squeeze_excitation",
            "placement": "after_relu_before_dropout",
            "squeeze": "global_average_pool_over_time",
        },
    },
}


BEST_ACCURACY_ECA_CONFIG = {
    "name": "best_accuracy_eca",
    "base_model": "best_accuracy",
    "preprocessed_data": BEST_ACCURACY_CONFIG["preprocessed_data"],
    "model": {
        **BEST_ACCURACY_CONFIG["model"],
        "attention": {
            "type": "eca_channel",
            "placement": "after_relu_before_dropout",
            "gamma": 2,
            "b": 1,
            "adaptive_kernel_size_for_C32": 3,
            "squeeze": "global_average_pool_over_time",
            "channel_interaction": "Conv1d across channel descriptor",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Best Accuracy 1D-SepCNN with paper LOSO/AOS protocols."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the extracted best-accuracy directory.",
    )
    parser.add_argument(
        "--model",
        choices=("best_accuracy", "best_accuracy_channel_temporal", "best_accuracy_se", "best_accuracy_eca"),
        default="best_accuracy",
        help="Model configuration. Default keeps the original baseline.",
    )
    parser.add_argument(
        "--protocol",
        choices=("loso", "aos", "both"),
        required=True,
        help="Evaluation protocol. 'both' runs LOSO then AOS.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, mps, ...",
    )
    parser.add_argument(
        "--output-dir",
        default="results_best_accuracy",
        help="Directory where fold histories, models and summaries are saved.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help=(
            "Optional subset of participant IDs (e.g. P001) for a smoke test. "
            "Omit for the paper's full 15-fold evaluation."
        ),
    )
    parser.add_argument(
        "--validate-all-samples",
        action="store_true",
        help="Open every .npz once at startup and validate shape/dtype/finiteness.",
    )
    return parser.parse_args()


def save_protocol_summary(results: list[dict], protocol_dir: Path) -> dict:
    accuracies = [float(result["final_test_accuracy"]) for result in results]
    macro_precisions = [
        float(result["macro_precision_diagnostic"]) for result in results
    ]

    summary = {
        "configuration_name": results[0]["configuration_name"],
        "protocol": results[0]["protocol"],
        "num_folds": len(results),
        "mean_test_accuracy": mean(accuracies),
        # Population std is the dispersion across the complete set of folds.
        "std_test_accuracy": pstdev(accuracies),
        "mean_macro_precision_diagnostic": mean(macro_precisions),
        "std_macro_precision_diagnostic": pstdev(macro_precisions),
        "folds": results,
    }

    with (protocol_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (protocol_dir / "fold_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "held_out_subject",
                "train_samples",
                "test_samples",
                "calibration_samples",
                "final_training_loss",
                "final_training_accuracy",
                "final_test_loss",
                "final_test_accuracy",
                "macro_precision_diagnostic",
                "num_trainable_parameters",
                "seed",
                "epoch_count",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})

    return summary


def run_protocol(
    protocol: str,
    records,
    fold_subjects: list[str],
    output_root: Path,
    training_config: TrainingConfig,
) -> dict:
    protocol_dir = output_root / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for fold_index, held_out_subject in enumerate(fold_subjects, start=1):
        if protocol == "loso":
            train_records, test_records = make_loso_split(
                records, held_out_subject=held_out_subject
            )
            calibration_session = None
        elif protocol == "aos":
            train_records, test_records = make_aos_split(
                records,
                held_out_subject=held_out_subject,
                calibration_session=CALIBRATION_SESSION,
            )
            calibration_session = CALIBRATION_SESSION
        else:
            raise ValueError(protocol)

        result = run_fold(
            protocol=protocol,
            fold_index=fold_index,
            num_folds=len(fold_subjects),
            held_out_subject=held_out_subject,
            train_records=train_records,
            test_records=test_records,
            output_dir=protocol_dir,
            config=training_config,
            calibration_session=calibration_session,
        )
        results.append(result)

    summary = save_protocol_summary(results, protocol_dir)

    print("\n" + "=" * 72)
    print(f"FINAL {protocol.upper()} RESULTS")
    print("=" * 72)
    for result in results:
        print(
            f"{result['held_out_subject']}: "
            f"Accuracy = {result['final_test_accuracy'] * 100:.2f}% | "
            f"Macro precision* = "
            f"{result['macro_precision_diagnostic'] * 100:.2f}%"
        )
    print("-" * 72)
    print(f"Mean Accuracy:        {summary['mean_test_accuracy'] * 100:.2f}%")
    print(f"Std Accuracy:         {summary['std_test_accuracy'] * 100:.2f}%")
    print(
        "Mean Macro Precision*: "
        f"{summary['mean_macro_precision_diagnostic'] * 100:.2f}%"
    )
    print(
        "Std Macro Precision*:  "
        f"{summary['std_macro_precision_diagnostic'] * 100:.2f}%"
    )
    print("* diagnostic; paper does not specify precision averaging")
    print("=" * 72)

    return summary


def main() -> None:
    args = parse_args()

    records = scan_dataset(args.data_root)
    validate_sample_arrays(records, check_all=args.validate_all_samples)
    summary = dataset_summary(records)

    print("Dataset detected:")
    print(f"  samples:      {summary['num_samples']}")
    print(f"  subjects:     {summary['num_subjects']} ({', '.join(summary['subjects'])})")
    print(f"  sessions:     {summary['num_sessions']} ({', '.join(summary['sessions'])})")
    print(f"  classes:      {summary['num_classes']} ({', '.join(summary['class_names'])})")
    print(f"  input shape:  {tuple(summary['input_shape'])}")
    print(f"  class counts: {summary['samples_per_gesture']}")
    print("  preprocessing: already applied; loader does not modify signal")

    all_subjects = subjects(records)
    if args.subjects:
        requested = args.subjects
        unknown = sorted(set(requested) - set(all_subjects))
        if unknown:
            raise ValueError(f"Unknown subject(s): {unknown}")
        fold_subjects = [subject for subject in all_subjects if subject in requested]
        print(
            "WARNING: --subjects was supplied, so this is a partial/smoke run, "
            "not the full paper 15-fold evaluation."
        )
    else:
        fold_subjects = all_subjects
        if len(fold_subjects) != 15:
            raise RuntimeError(
                f"Full paper evaluation expects 15 subjects; found {len(fold_subjects)}."
            )

    if args.channel_reduction_ratio <= 0:
        raise ValueError("--channel-reduction-ratio must be positive")
    if args.se_reduction_ratio <= 0:
        raise ValueError("--se-reduction-ratio must be positive")

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        model_name=args.model,
        channel_reduction_ratio=args.channel_reduction_ratio,
        se_reduction_ratio=args.se_reduction_ratio,
    )

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configuration = (
        BEST_ACCURACY_CONFIG
        if args.model == "best_accuracy"
        else (
            BEST_ACCURACY_CHANNEL_TEMPORAL_CONFIG
            if args.model == "best_accuracy_channel_temporal"
            else (
                BEST_ACCURACY_SE_CONFIG
                if args.model == "best_accuracy_se"
                else BEST_ACCURACY_ECA_CONFIG
            )
        )
    )
    if args.model in {"best_accuracy_channel_temporal", "best_accuracy_se"}:
        configuration = json.loads(json.dumps(configuration))
        if args.model == "best_accuracy_channel_temporal":
            configuration["model"]["attention"]["channel_reduction_ratio"] = training_config.channel_reduction_ratio
        else:
            configuration["model"]["attention"]["reduction_ratio"] = training_config.se_reduction_ratio

    with (output_root / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "configuration": configuration,
                "training": {
                    "epochs": training_config.epochs,
                    "batch_size": training_config.batch_size,
                    "learning_rate": training_config.learning_rate,
                    "seed": training_config.seed,
                    "num_workers": training_config.num_workers,
                    "device": training_config.device,
                    "model_name": training_config.model_name,
                    "channel_reduction_ratio": training_config.channel_reduction_ratio,
                    "se_reduction_ratio": training_config.se_reduction_ratio,
                },
                "dataset": summary,
                "fold_subjects": fold_subjects,
            },
            handle,
            indent=2,
        )

    protocols = ("loso", "aos") if args.protocol == "both" else (args.protocol,)

    for protocol in protocols:
        run_protocol(
            protocol=protocol,
            records=records,
            fold_subjects=fold_subjects,
            output_root=output_root,
            training_config=training_config,
        )


if __name__ == "__main__":
    main()
