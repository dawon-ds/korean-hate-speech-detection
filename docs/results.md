# Experiment Results

The following values are preserved from the final project materials.

| Setting | Coarse Macro F1 | Coarse Micro F1 | Coarse LRAP | Fine Macro F1 | Fine Micro F1 | Fine LRAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat | 0.7395 | 0.7188 | 0.8584 | 0.7073 | 0.6915 | 0.9603 |
| Flat + Augmentation | 0.7299 | 0.7094 | 0.8519 | 0.7085 | 0.6976 | — |
| Hierarchical | — | — | — | 0.7258 | 0.6552 | 0.9560 |
| Hierarchical + Augmentation | 0.7415 | 0.7163 | 0.8577 | 0.7292 | 0.6610 | 0.9569 |

## Interpretation

- The hierarchical + augmentation run achieved the strongest recorded **fine Macro F1: 0.7292**.
- Flat classification achieved higher recorded **fine Micro F1** and **fine LRAP**.
- Augmentation did not improve every metric consistently.
- HITL neutral-data experiments also did not produce a clear overall F1/LRAP gain in the final materials.

Missing values are left blank rather than inferred.
