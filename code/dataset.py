from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = (
    "knock",
    "swipe-down",
    "swipe-left",
    "swipe-right",
    "swipe-up",
    "tap",
)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

EXPECTED_INPUT_CHANNELS = 4
EXPECTED_WINDOW_SAMPLES = 1250
CALIBRATION_SESSION = "S001"


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    subject: str
    session: str
    gesture: str
    event: str


def _parse_event_id(path: Path) -> str:
    marker = "_event-"
    stem = path.stem
    if marker not in stem:
        raise ValueError(f"Cannot parse event id from filename: {path.name}")
    return stem.split(marker, 1)[1]


def scan_dataset(data_root: str | Path) -> list[SampleRecord]:
    """Scan the already-preprocessed Best Accuracy dataset.

    Expected layout:
        data_root/sub-PXXX/ses-SXXX/<gesture>/*.npz

    No filtering, downsampling, window extraction, resampling, or normalization
    is performed here.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    records: list[SampleRecord] = []

    for path in sorted(root.glob("sub-P*/ses-S*/*/*.npz")):
        # Ignore macOS AppleDouble metadata if an extracted archive contains it.
        if path.name.startswith("._") or "__MACOSX" in path.parts:
            continue

        gesture = path.parent.name
        session_dir = path.parent.parent.name
        subject_dir = path.parent.parent.parent.name

        if gesture not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unexpected gesture directory '{gesture}' in {path}. "
                f"Expected one of {CLASS_NAMES}."
            )
        if not subject_dir.startswith("sub-"):
            raise ValueError(f"Unexpected subject directory: {subject_dir}")
        if not session_dir.startswith("ses-"):
            raise ValueError(f"Unexpected session directory: {session_dir}")

        records.append(
            SampleRecord(
                path=path,
                subject=subject_dir.removeprefix("sub-"),
                session=session_dir.removeprefix("ses-"),
                gesture=gesture,
                event=_parse_event_id(path),
            )
        )

    if not records:
        raise RuntimeError(
            f"No real .npz samples found under {root}. "
            "Expected sub-P*/ses-S*/*/*.npz."
        )

    return records


def validate_sample_arrays(records: Sequence[SampleRecord], check_all: bool = True) -> None:
    """Validate that stored samples are model-ready [channels, time] arrays.

    This function only validates; it never changes the signal.
    """
    to_check = records if check_all else records[:1]

    for record in to_check:
        with np.load(record.path, allow_pickle=False) as data:
            if "x" not in data.files:
                raise KeyError(f"Missing array key 'x' in {record.path}")
            x = data["x"]

        if x.shape != (EXPECTED_INPUT_CHANNELS, EXPECTED_WINDOW_SAMPLES):
            raise ValueError(
                f"Unexpected shape {x.shape} in {record.path}; expected "
                f"({EXPECTED_INPUT_CHANNELS}, {EXPECTED_WINDOW_SAMPLES})."
            )
        if x.dtype != np.float32:
            raise TypeError(
                f"Unexpected dtype {x.dtype} in {record.path}; expected float32."
            )
        if not np.isfinite(x).all():
            raise ValueError(f"NaN/Inf detected in {record.path}")


def subjects(records: Iterable[SampleRecord]) -> list[str]:
    return sorted({record.subject for record in records})


def dataset_summary(records: Sequence[SampleRecord]) -> dict:
    by_subject = Counter(record.subject for record in records)
    by_session = Counter(record.session for record in records)
    by_gesture = Counter(record.gesture for record in records)

    return {
        "num_samples": len(records),
        "subjects": subjects(records),
        "num_subjects": len(by_subject),
        "sessions": sorted(by_session),
        "num_sessions": len(by_session),
        "class_names": list(CLASS_NAMES),
        "num_classes": len(CLASS_NAMES),
        "samples_per_subject": dict(sorted(by_subject.items())),
        "samples_per_gesture": {name: by_gesture[name] for name in CLASS_NAMES},
        "input_shape": [EXPECTED_INPUT_CHANNELS, EXPECTED_WINDOW_SAMPLES],
    }


def make_loso_split(
    records: Sequence[SampleRecord], held_out_subject: str
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Paper LOSO: all sessions of one participant are test; all others train."""
    train_records = [r for r in records if r.subject != held_out_subject]
    test_records = [r for r in records if r.subject == held_out_subject]

    if not test_records:
        raise ValueError(f"Unknown held-out subject: {held_out_subject}")
    if any(r.subject == held_out_subject for r in train_records):
        raise AssertionError("LOSO leakage: held-out subject found in training set.")
    if any(r.subject != held_out_subject for r in test_records):
        raise AssertionError("LOSO split error: test set contains another subject.")

    return train_records, test_records


def make_aos_split(
    records: Sequence[SampleRecord],
    held_out_subject: str,
    calibration_session: str = CALIBRATION_SESSION,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Paper AOS: LOSO + held-out subject's first session in training.

    The held-out participant's S001 is the calibration session. Their remaining
    sessions (S002-S010 in this dataset) are the test set.
    """
    subject_records = [r for r in records if r.subject == held_out_subject]
    if not subject_records:
        raise ValueError(f"Unknown held-out subject: {held_out_subject}")

    train_records = [
        r
        for r in records
        if r.subject != held_out_subject
        or (r.subject == held_out_subject and r.session == calibration_session)
    ]
    test_records = [
        r
        for r in records
        if r.subject == held_out_subject and r.session != calibration_session
    ]

    calibration_records = [
        r
        for r in train_records
        if r.subject == held_out_subject and r.session == calibration_session
    ]

    if not calibration_records:
        raise RuntimeError(
            f"AOS requires calibration session {calibration_session} for "
            f"{held_out_subject}, but no samples were found."
        )
    if not test_records:
        raise RuntimeError(
            f"AOS test set is empty for {held_out_subject} after reserving "
            f"{calibration_session} for calibration."
        )
    if any(
        r.subject == held_out_subject and r.session != calibration_session
        for r in train_records
    ):
        raise AssertionError("AOS leakage: held-out test sessions found in training.")
    if any(
        r.subject != held_out_subject or r.session == calibration_session
        for r in test_records
    ):
        raise AssertionError("AOS split error in held-out subject/session assignment.")

    return train_records, test_records


class GestureDataset(Dataset):
    """Load already-preprocessed `.npz` windows without signal preprocessing."""

    def __init__(self, records: Sequence[SampleRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]

        with np.load(record.path, allow_pickle=False) as data:
            x = data["x"]

        if x.shape != (EXPECTED_INPUT_CHANNELS, EXPECTED_WINDOW_SAMPLES):
            raise ValueError(
                f"Unexpected shape {x.shape} in {record.path}; expected "
                f"({EXPECTED_INPUT_CHANNELS}, {EXPECTED_WINDOW_SAMPLES})."
            )

        # np.asarray ensures a normal contiguous ndarray even if the archive
        # representation changes. No normalization or other signal operation.
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32).copy())
        y_tensor = torch.tensor(CLASS_TO_INDEX[record.gesture], dtype=torch.long)

        return x_tensor, y_tensor
