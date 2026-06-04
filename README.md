# Russian Spellcheck Experiments

Research project on Russian spell correction using decoder-only Large Language Models and a novel dataset containing device-type annotations.

## Overview

This project investigates whether a decoder-only language model can be effectively fine-tuned for Russian spell correction and compete with state-of-the-art encoder-decoder approaches such as SAGE from SberDevices.

During the research, a major limitation of existing Russian spellcheck datasets was identified: they do not contain information about the device used to enter text (desktop, mobile, or tablet). Since typing behavior and error patterns differ significantly across devices, a new dataset was created to address this issue.

The project includes:

- Fine-tuning of a decoder-only model for Russian spell correction
- Collection of a novel dataset with device labels
- Development of a data collection application
- Experiments on spell correction and device classification

## Motivation

Most publicly available Russian spellcheck datasets are collected from existing internet sources and therefore lack metadata about how the text was entered.

However, typing errors vary depending on the device:

- Mobile devices introduce touchscreen-related mistakes
- Desktop keyboards produce different typo patterns
- Tablets combine characteristics of both

The goal of this work is to investigate whether device-aware spell correction can improve future spellcheck systems.

## Dataset

### RU_SPELLCHECK_DEVICE

https://huggingface.co/datasets/BW/RU_SPELLCHECK_DEVICE

A new dataset was collected using a custom application where users were asked to retype sentences from Russian classical literature. https://github.com/BorisVIkVik/SpellCheck_Game_Dataset

For every sample, the application stores:

- Original sentence
- User-typed sentence
- Anonymous user identifier (`player_id`)
- Timestamp
- Device type (`desktop`, `mobile`, `tablet`)

### Statistics

| Metric | Value |
|----------|----------|
| Train samples | 269 |
| Test samples | 269 |
| Device classes | 3 |
| Unique users | 33 |

## Model

### Spell Correction Model

Base model:

- Qwen3.5-0.8B

Training method:

- Supervised Fine-Tuning (SFT)
- Instruction tuning
- Unsloth framework

Training environment:

- Google Colab

## Results on RU_SPELLCHECK_DEVICE dataset

### Fine-Tuned Qwen3.5-0.8B

https://huggingface.co/BW/Qwen3.5_Fine-tuned_on_RU_SPELLCHECK_DEVICE

| Metric Type | Precision | Recall | F1 |
|-------------|------------|---------|-----|
| Overall | 35.17 | 31.28 | 33.11 |
| SPELL | 39.60 | 37.02 | 38.27 |
| CASE | 47.73 | 50.00 | 48.84 |
| PUNCT | 53.85 | 50.30 | 52.01 |
| YO | 0.00 | 0.00 | 0.00 |

### SAGE-FREDT5-Large (SberDevices)

| Metric Type | Precision (%) | Recall (%) | F1 (%) |
|-------------|---------------|------------|---------|
| Overall     | 64.42         | 73.73      | 68.76   |
| SPELL       | 59.91         | 65.42      | 62.55   |
| CASE        | 17.24         | 47.62      | 25.32   |
| PUNCT       | 11.38         | 25.75      | 15.78   |
| YO          | 0.00          | 0.00       | 0.00    |


So we can see that my model overcame SAGE-FREDT5-Large in case and punctuation correction.

## Honorable Mention

New dataset give interesting opportunitues in text classification task, because
we can try to predict the device that the person uses. This maybe useful for
improving spellcheck because typing on different device causes different error.
Also in maybe useful in Criminalistics. Here are results of simple test. I’ve
fine-tuned DeepPavlov/rubert-base-cased for predicting device label using typed
text.

| Metric                     | Value      |
|---------------------------|-----------:|
| eval_loss                 | 0.73699    |
| eval_accuracy             | 0.68919    |
| eval_f1                   | 0.68411    |
| eval_precision            | 0.69390    |
| eval_recall               | 0.68919    |
| eval_runtime (s)          | 2.6592     |
| eval_samples_per_second   | 111.313    |
| eval_steps_per_second     | 7.145      |
| epoch                     | 5.0        |
## Citation

```bibtex
@misc{viktorov2026spellcheck,
  author = {Boris Viktorov},
  title = {Supervised Fine-Tuning of Decoder-Only Model for Russian Spellcheck Task},
  year = {2026}
}
```
