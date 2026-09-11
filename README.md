# Korean Hate Speech Detection

**Multi-label Korean NLP Classification with PLM Comparison, Lexicon Augmentation, and Hierarchical Modeling**

This repository is a cleaned portfolio version of a team project on **Korean hate-speech classification**. The project compares Korean pretrained language models, studies the effect of HITL and lexicon-based augmentation, and evaluates both flat and hierarchical classification settings.

> **Task:** Korean online text → coarse toxicity class + fine-grained hate categories

## Project Overview

- **Period:** 2025
- **Task:** Korean hate-speech / abusive-language classification
- **Datasets:** UnSmile + HateScore
- **Models explored:** BERT, ELECTRA, RoBERTa
- **Framework:** PyTorch + Hugging Face Transformers
- **Evaluation:** Macro F1, Micro F1, LRAP
- **Additional work:** HITL data experiment, Hybrid Lexicon augmentation, Flat vs Hierarchical modeling, Web Demo

## Problem

Korean hate speech is difficult to classify because harmful expressions can be indirect, obfuscated, context-dependent, or associated with multiple target groups at once. The project therefore explored more than a single baseline classifier:

1. compare Korean pretrained language models,
2. test additional HITL data,
3. construct a hybrid hate-expression lexicon,
4. apply robustness-oriented text augmentation,
5. compare flat and hierarchical prediction structures,
6. build a simple abusive-chat filtering demo.

## Dataset

The experiments used two Korean hate-speech datasets:

- **UnSmile**
- **HateScore**

The baseline pipeline supports training with UnSmile alone or with a combined UnSmile + HateScore training set. Large raw datasets are intentionally **not included** in this public repository.

## Baseline PLM Experiments

The baseline code compares Korean pretrained language models such as:

- `klue/bert-base`
- `monologg/koelectra-base-v3-discriminator`
- `klue/roberta-base`

An additional ALBERT experiment also remained in the development code, although the final report/presentation focused on the main Korean PLM comparisons.

## HITL Experiment

Human-in-the-loop neutral data was added to examine whether more ambiguous non-hate examples would improve the decision boundary. The project logs did **not** show a clear overall F1/LRAP improvement, although inference examples suggested possible benefits for some ambiguous neutral/hate cases.

## Hybrid Lexicon & Robustness Augmentation

The augmentation pipeline combined:

- manually curated hate-expression lexicon entries,
- automatically mined seed terms,
- deduplication and filtering,
- character repetition variants,
- Hangul/Jamo decomposition,
- numeric substitutions,
- prefix/suffix variations and other obfuscation patterns.

The goal was not simply to increase dataset size, but to make the classifier more robust to the kinds of spelling variation and obfuscation often found in online text.

> The full lexicon is omitted from this portfolio repository. The original team code contains the complete augmentation implementation and lexical resources.

## Flat vs Hierarchical Classification

The final modeling stage compared two structures.

### Flat model

- coarse prediction head
- fine-grained multi-label prediction head
- combined coarse CE loss + fine BCE loss

### Hierarchical model

The hierarchical model adds a consistency penalty between coarse and fine predictions.

For example:

- a sample predicted as **clean** should not simultaneously activate fine hate categories,
- a sample predicted as **offensive/hate** should not return an empty fine-label set.

## Results

### Main comparison

| Setting | Coarse Macro F1 | Coarse Micro F1 | Coarse LRAP | Fine Macro F1 | Fine Micro F1 | Fine LRAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat | 0.7395 | 0.7188 | 0.8584 | 0.7073 | 0.6915 | 0.9603 |
| Flat + Augmentation | 0.7299 | 0.7094 | 0.8519 | 0.7085 | 0.6976 | — |
| Hierarchical | — | — | — | 0.7258 | 0.6552 | 0.9560 |
| Hierarchical + Augmentation | **0.7415** | 0.7163 | 0.8577 | **0.7292** | 0.6610 | 0.9569 |

Missing values are left blank rather than inferred.

### Interpretation

- Augmentation did **not** improve every metric consistently.
- The hierarchical + augmentation setting produced the strongest recorded **fine Macro F1 (0.7292)**.
- Flat classification retained stronger recorded **fine Micro F1 / LRAP**.
- The project therefore treats augmentation and hierarchy as trade-offs rather than claiming a universal improvement.

## Repository Structure

```text
korean-hate-speech-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── base.yaml
├── src/
│   ├── dataset.py
│   ├── metrics.py
│   ├── models.py
│   ├── trainer.py
│   └── infer.py
└── docs/
    ├── augmentation.md
    └── results.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

The cleaned training pipeline expects columns corresponding to:

```text
text, hate_label, gender, LGBT, age, region, race, religion, socioeconomic, etc
```

The exact preprocessing depends on the source dataset and is intentionally kept outside the public portfolio repository because the original project used externally sourced datasets.

## Notes

- This is a **team project** and the repository documents the overall technical pipeline.
- The original report/presentation did not specify member-level implementation ownership, so this repository does not attribute every component to the repository owner individually.
- Large datasets, checkpoints, logs, and the full lexical resource are excluded.
- Some development code included experiments that were later removed from the final report; this portfolio version focuses on the final project narrative and verified results.

## Tech Stack

`Python` `PyTorch` `Transformers` `scikit-learn` `Pandas` `BERT` `ELECTRA` `RoBERTa` `Multi-label Classification` `NLP`
