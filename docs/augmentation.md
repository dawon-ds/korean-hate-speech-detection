# Augmentation Pipeline

The original team implementation used a hybrid lexical resource for robustness-oriented augmentation.

## Lexicon construction

1. Manual seed lexicon
2. Automatically mined seed terms
3. Merge
4. Deduplicate / filter
5. Build enhanced lexicon

## Perturbation types

The project materials document several obfuscation patterns frequently observed in Korean online text:

- character repetition
- Hangul/Jamo decomposition
- numeric substitution
- prefix/suffix variation
- surface-form variation around lexical entries

Augmentation was mainly applied to hate-labeled training examples. The final public portfolio repository does not redistribute the full lexical list or the raw augmented dataset.

## Why augmentation was tested

The goal was robustness rather than simple dataset expansion. Hate expressions can be intentionally altered to evade filters, so the experiments tested whether exposure to perturbed spellings could improve fine-grained classification.

The recorded results show mixed effects: augmentation improved some fine-level metrics while decreasing others, so the project does not claim uniform performance gains.
