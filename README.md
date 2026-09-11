# Korean Hate Speech Detection

**Multi-label Korean NLP Classification with PLM Comparison, Lexicon Augmentation, and Hierarchical Modeling**

This project explores Korean hate-speech classification through pretrained language model comparison, dataset integration, robustness-oriented text augmentation, and flat vs. hierarchical classification.

> **Task:** Korean online text → coarse toxicity class + fine-grained hate categories

## Project Overview

- **Period:** 2025
- **Task:** Korean hate-speech / abusive-language classification
- **Datasets:** UnSmile + HateScore
- **Models:** BERT, ELECTRA, RoBERTa
- **Framework:** PyTorch + Hugging Face Transformers
- **Evaluation:** Macro F1, Micro F1, LRAP
- **Experiments:** PLM comparison, HITL, Hybrid Lexicon augmentation, Flat vs. Hierarchical modeling

## Problem

Korean hate speech is difficult to classify because harmful expressions can be indirect, obfuscated, context-dependent, or associated with multiple target groups. The project therefore explored several directions beyond a single baseline classifier:

1. comparison of Korean pretrained language models,
2. integration of UnSmile and HateScore data,
3. HITL neutral-data experiments,
4. construction of a hybrid hate-expression lexicon,
5. robustness-oriented text augmentation,
6. comparison of flat and hierarchical prediction structures.

## Dataset

The experiments used two Korean hate-speech datasets:

- **UnSmile**
- **HateScore**

The baseline pipeline supports either UnSmile-only training or a combined UnSmile + HateScore training set. `build_combined_dataset.py` maps HateScore labels to the UnSmile schema and creates the combined training data.

Large raw datasets are not included in this repository. Place the required dataset files under the corresponding experiment `data/` directory before running the scripts.

### Baseline data

The baseline experiment uses files such as:

```text
unsmile_train.csv
unsmile_valid.csv
hatescore.csv
train_hatescore_unsmile.csv
```

The training code uses the Korean text column `문장` and a 10-class label mapping consisting of clean, abusive language, and hate-target categories.

### Augmentation data

The augmentation experiment expects:

```text
merged_dataset_v1.1.csv
```

under `experiments/augmentation/data/`, together with the lexical resources required by the augmentation pipeline when applicable.

## Baseline PLM Experiments

The baseline experiment compares Korean pretrained language models including:

- `klue/bert-base`
- `monologg/koelectra-base-v3-discriminator`
- `klue/roberta-base`

An ALBERT configuration is also retained in the experiment code. The training script evaluates combinations of model architecture and hyperparameters and records test metrics and training logs.

The default baseline configuration is:

```text
experiments/baseline/config/text_classification.yaml
```

and the main training entry point is:

```text
experiments/baseline/scripts/train_baseline.py
```

## HITL Experiment

Human-in-the-loop neutral data was added to examine whether more ambiguous non-hate examples would improve the decision boundary. The recorded experiments did **not** show a clear overall F1/LRAP improvement, although inference examples suggested possible benefits for some ambiguous neutral/hate cases.

## Hybrid Lexicon & Robustness Augmentation

The augmentation pipeline combines:

- manually curated hate-expression lexicon entries,
- automatically mined seed terms,
- deduplication and filtering,
- character repetition variants,
- Hangul/Jamo decomposition,
- numeric substitutions,
- prefix/suffix variations and other obfuscation patterns.

The goal is to improve robustness to spelling variation and intentional obfuscation commonly found in online text. The implementation is available in `experiments/augmentation/src/augment.py`; large lexical resource files are excluded from the repository.

## Flat vs. Hierarchical Classification

The final modeling stage compares two structures.

### Flat model

- coarse prediction head
- fine-grained multi-label prediction head
- combined coarse CE loss + fine BCE loss

### Hierarchical model

The hierarchical model adds a differentiable consistency loss between coarse and fine prediction probabilities. It penalizes cases where the model assigns a high clean probability while also activating fine hate categories, as well as cases where toxic coarse predictions are paired with no fine hate category.

Both flat and hierarchical training scripts select the best validation checkpoint by fine-grained Macro F1 and apply early stopping using the configured patience value.

The corresponding training entry points are:

```text
experiments/augmentation/train_flat.py
experiments/augmentation/train_hier.py
```

## Results

| Setting | Coarse Macro F1 | Coarse Micro F1 | Coarse LRAP | Fine Macro F1 | Fine Micro F1 | Fine LRAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat | 0.7395 | 0.7188 | 0.8584 | 0.7073 | 0.6915 | 0.9603 |
| Flat + Augmentation | 0.7299 | 0.7094 | 0.8519 | 0.7085 | 0.6976 | — |
| Hierarchical | — | — | — | 0.7258 | 0.6552 | 0.9560 |
| Hierarchical + Augmentation | **0.7415** | 0.7163 | 0.8577 | **0.7292** | 0.6610 | 0.9569 |

### Interpretation

- Augmentation did **not** improve every metric consistently.
- Hierarchical + augmentation produced the strongest recorded **fine Macro F1 (0.7292)**.
- Flat classification retained stronger recorded **fine Micro F1 / LRAP**.
- The results therefore indicate trade-offs between augmentation, hierarchical consistency, and different evaluation metrics rather than a universal improvement.

The table above reflects the recorded project experiment results. The public repository has since received code-quality and training-stability fixes, so rerunning the current code may not reproduce the historical values exactly.

## Repository Structure

```text
korean-hate-speech-detection/
├── README.md
├── requirements.txt
├── config/
│   └── base.yaml
├── src/
│   ├── dataset.py
│   ├── metrics.py
│   ├── models.py
│   ├── trainer.py
│   └── infer.py
├── docs/
│   ├── augmentation.md
│   └── results.md
└── experiments/
    ├── baseline/
    │   ├── config/
    │   │   └── text_classification.yaml
    │   ├── scripts/
    │   │   ├── train_baseline.py
    │   │   └── inference.py
    │   ├── src/
    │   │   └── dataset_text_only.py
    │   └── utils/
    │       ├── build_combined_dataset.py
    │       ├── unsmile_train_valid_split.py
    │       └── utils.py
    └── augmentation/
        ├── config/
        │   └── base.yaml
        ├── train_flat.py
        ├── train_hier.py
        ├── infer.py
        └── src/
            ├── augment.py
            ├── config.py
            ├── dataset.py
            ├── metrics.py
            ├── models.py
            ├── trainer.py
            └── utils.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Experiments

Prepare the required datasets in each experiment's `data/` directory first.

For the baseline pipeline, build the combined UnSmile + HateScore dataset when needed and then run the PLM experiments:

```bash
python experiments/baseline/utils/build_combined_dataset.py
python experiments/baseline/scripts/train_baseline.py
```

For the augmentation experiments:

```bash
python experiments/augmentation/train_flat.py
python experiments/augmentation/train_hier.py
```

The augmentation scripts resolve their configuration, dataset, log, and checkpoint paths relative to `experiments/augmentation/`, so the commands above can be run from the repository root.

The augmentation configuration currently defaults to CUDA. Change `device: "cuda"` to `device: "cpu"` in `experiments/augmentation/config/base.yaml` when running without a CUDA-capable GPU.

## Notes

- Large datasets, model checkpoints, training logs, and large lexical resource files are excluded from the repository.
- The repository includes preprocessing, baseline PLM experiments, augmentation, flat/hierarchical training, evaluation, and inference code used across the project experiments.
- Some exploratory model configurations remain in the code even when they were not emphasized in the final comparison.

## Tech Stack

`Python` `PyTorch` `Transformers` `scikit-learn` `Pandas` `BERT` `ELECTRA` `RoBERTa` `Multi-label Classification` `NLP`
