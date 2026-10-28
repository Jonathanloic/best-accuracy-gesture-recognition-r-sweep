# Attention Mechanisms for Surface Vibration-based Gesture Recognition

## Best Accuracy Configuration

This repository contains the code for evaluating four lightweight attention mechanisms on the **Best Accuracy** backbone (8,722 parameters, 6 blocks, 32 channels, kernel size 15) from the surface-vibration gesture recognition system of [Hettstedt et al. (2026)](https://arxiv.org/abs/2605.10110).

This work is part of a Bachelor thesis at the **Universität Duisburg-Essen**, Intelligent Embedded Systems Group.

> **Companion repository:** The [Least Parameters configuration](https://github.com/Jonathanloic/least-parameter-gesture-recognition-r-sweep) (1,802 parameters, 4 blocks, 16 channels, kernel size 9) is evaluated in a separate repository with the same structure.

---

## Attention Variants

| Variant | Mechanism | Parameters Added | Hyperparameter |
|---|---|---|---|
| Baseline | None | 0 | — |
| SE | Channel (bottleneck) | 768–6,144 | r ∈ {2, 4, 8, 16} |
| ECA | Channel (local conv) | 18 | k = 3 (derived) |
| Temporal | Temporal (conv over time) | 84 | k = 7 |
| Channel–Temporal | Channel then temporal | 852–6,228 | r ∈ {2, 4, 8, 16}, k = 7 |

All modules are inserted after ReLU and before Dropout in every depthwise-separable block.

---

## Repository Structure

```
├── model.py                          # Baseline depthwise-separable 1D-CNN
├── model_se.py                       # Squeeze-and-Excitation module
├── model_eca.py                      # Efficient Channel Attention module
├── model_temporal.py                 # Temporal attention module
├── model_attention.py                # Channel–temporal attention module
├── dataset.py                        # Data loading and preprocessing
├── train.py                          # Training loop
├── experiments.py                    # Single LOSO evaluation run
├── repeat_experiments.py             # 20-run orchestration script
├── reproducibility.py                # Seeding and deterministic configuration
├── sweep_attention_reduction_ratio.py # Reduction ratio sweep launcher
├── verify_temporal_model.py          # Temporal model verification
└── results/                          # Experimental results (JSON summaries)
```

---

## Evaluation Protocol

- **Cross-validation:** Leave-one-subject-out (LOSO), 15 folds
- **Repetitions:** 20 independent runs (seeds 42–61)
- **Epochs:** 300 (fixed, no early stopping)
- **Optimiser:** AdamW, lr = 1×10⁻³, weight decay = 0.01 (default)
- **Batch size:** 128
- **Loss:** Cross-entropy, unweighted
- **Determinism:** cuDNN deterministic enabled, autotuning disabled

Every reported result is the mean ± population standard deviation over the 20 runs.

---

## Usage

### Single Evaluation Run

```bash
python experiments.py \
    --model best_accuracy_se \
    --protocol loso \
    --data-root /path/to/preprocessed-data \
    --output-dir results/se_r2 \
    --seed 42 \
    --epochs 300 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --se-reduction-ratio 2
```

### Full 20-Run Evaluation

```bash
python repeat_experiments.py \
    --model best_accuracy_se \
    --protocol loso \
    --data-root /path/to/preprocessed-data \
    --output-dir results/se_r2 \
    --runs 20 \
    --base-seed 42 \
    --se-reduction-ratio 2
```

### Reduction Ratio Sweep

```bash
python sweep_attention_reduction_ratio.py \
    --model best_accuracy_se \
    --data-root /path/to/preprocessed-data \
    --output-dir results/
```

This runs the full 20-run evaluation for r ∈ {2, 4, 8, 16} sequentially.

---

## Key Results

| Variant | Clean (%) | Noisy (%) | Parameters |
|---|---|---|---|
| Baseline | 91.23 ± 0.71 | 71.22 ± 0.56 | 8,722 |
| **SE (r=2)** | **91.85 ± 0.97** | 72.40 ± 0.60 | 14,866 |
| ECA | 91.76 ± 0.87 | **72.72 ± 0.59** | **8,740** |
| Temporal | 90.91 ± 1.00 | 71.67 ± 0.60 | 8,806 |
| CT (r=8) | 91.63 ± 0.81 | 71.46 ± 0.51 | 10,342 |

SE leads on clean data. ECA leads under noise at the lowest parameter cost (+18 parameters). Temporal attention falls below the baseline on clean data.

---

## Requirements

- Python 3.10+
- PyTorch (with CUDA support)
- NumPy

---

## Dataset

The preprocessed surface-vibration gesture corpus is from [Hettstedt et al. (2026)](https://arxiv.org/abs/2605.10110). It contains 8,999 gesture events from 15 participants across 6 classes (swipe-left, swipe-right, swipe-up, swipe-down, tap, knock), recorded with 4 piezoelectric sensors at 1 kHz (no downsampling for this configuration).

**The dataset is not included in this repository.** Contact the [Intelligent Embedded Systems Group](https://www.2is.uni-due.de/) at the Universität Duisburg-Essen for access.

---

## Citation

If you use this code, please cite:

```bibtex
@thesis{fondjo2026attention,
  author  = {Fondjo Kamtchueng, Jonathan Loic},
  title   = {Attention Mechanisms for Surface Vibration-based
             Gesture Recognition},
  school  = {Universität Duisburg-Essen},
  year    = {2026},
  type    = {Bachelor Thesis}
}
```

---

## License

This project was developed as part of an academic thesis. Please contact the author for usage terms.
