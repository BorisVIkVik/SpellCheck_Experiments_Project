# Supervised fine-tuning of decoder-only model for Russian spellcheck task
---
Research project on Russian spell correction using decoder-only Large Language Models and a novel dataset containing device-type annotations.

## Overview
---
This project investigates whether a decoder-only language model can be effectively fine-tuned for Russian spell correction and compete with state-of-the-art encoder-decoder approaches such as SAGE from SberDevices.

During the research, a major limitation of existing Russian spellcheck datasets was identified: they do not contain information about the device used to enter text (desktop, mobile, or tablet). Since typing behavior and error patterns differ significantly across devices, a new dataset was created to address this issue.

The project includes:

- Fine-tuning of a decoder-only models for Russian spell correction
- Collection of a novel dataset with device labels
- Development of a data collection application
- Experiments on spell correction and device classification



## Motivation
---
Most publicly available Russian spellcheck datasets are collected from existing internet sources and therefore lack metadata about how the text was entered.

However, typing errors vary depending on the device:

- Mobile devices introduce touchscreen-related mistakes
- Desktop keyboards produce different typo patterns
- Tablets combine characteristics of both

The goal of this work is to investigate whether device-aware spell correction can improve future spellcheck systems.

## Dataset
---
### RU_SPELLCHECK_DEVICE

[https://huggingface.co/datasets/BW/RU_SPELLCHECK_DEVICE](https://huggingface.co/datasets/BW/RU_SPELLCHECK_DEVICE)

A new dataset was collected using a custom application where users were asked to retype sentences from Russian classical literature. [https://github.com/BorisVIkVik/SpellCheck_Game_Dataset](https://github.com/BorisVIkVik/SpellCheck_Game_Dataset)

For every sample, the application stores:

- Original sentence
- User-typed sentence
- Anonymous user identifier (`player_id`)
- Timestamp
- Device type (`desktop`, `mobile`, `tablet`)



### Statistics


| Metric         | Value |
| -------------- | ----- |
| Train samples  | 269   |
| Test samples   | 269   |
| Device classes | 3     |
| Unique users   | 33    |




## Experiments
---
### Base models:
- [mistralai/Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)
- [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
### Training method:
- [Genetic-Pareto (GEPA)](https://github.com/gepa-ai/gepa)
- Supervised Fine-Tuning (Instruction tuning) 
### Training environment:
- Google Colab
- Cloud.ru


## Results
---
I've used GEPA to optimize prompt. The result is this prompt:

    You are a Russian‑language proofreader.  
    For every input you receive, perform **only** the following actions:
    1. **Correct** all orthographic (spelling), punctuation, and register/case mistakes.  
      - Fix misspelled words, missing or extra letters, and wrong word forms.  
      - Use proper Russian punctuation: commas, periods, question/exclamation marks, em‑dashes (—) for dialogue, hyphens (‑) only where a hyphen is required, and matching quotation marks.  
      - Ensure correct capitalization: the first word of a sentence (or after an em‑dash that starts a new sentence) must start with a capital letter; proper nouns must be capitalized; other words should be lower‑case unless the context demands otherwise.  
      - Preserve the original meaning; do not add, delete, or replace words unless it is necessary to fix an error.

    2. **Output** **exactly** the corrected text, **nothing else**:
      - No explanations, reasoning, or meta‑information.  
      - No tags such as `<think>` or any other markup.  
      - Do **not** surround the result with quotation marks.  
      - Keep the original line breaks (if any) and spacing, adjusting only what is required for correct punctuation (e.g., a single space after a comma, period, dash, etc.).  
      - If the input is already correct, output it unchanged.

    **Examples**

    - Input: `Дают приятрыз дух, увеселяют взор И вам якрасавицы, хранят чебя в цбор.`  
      Output: `Дают приятный дух, увеселяют взор, и вам, красавицы, хранят себя в убор.`

    - Input: `- Чей эоо дом? - спросил он у будувого будочника.`  
      Output: `— Чей это дом? — спросил он у углового будочника.`

    - Input: `Тихий ветер шевелмл листья старого дерева.`  
      Output: `Тихий ветер шевелил листья старого дерева.`

    Follow these rules for every request.

Trained models:
- [BW/Qwen2.5-14b-Instruct-RU-Spellcheck-fine-tuned](https://huggingface.co/BW/Qwen2.5-14b-Instruct-RU-Spellcheck-fine-tuned)
- [BW/Qwen2.5-7b-Instruct-RU-Spellcheck-fine-tuned](https://huggingface.co/BW/Qwen2.5-7b-Instruct-RU-Spellcheck-fine-tuned)
- [BW/Ministral-8b-Instruct-RU-Spellcheck-fine-tuned](https://huggingface.co/BW/Ministral-8b-Instruct-RU-Spellcheck-fine-tuned)

### My Best Model: [BW/Qwen2.5-14b-Instruct-RU-Spellcheck-fine-tuned](https://huggingface.co/BW/Qwen2.5-14b-Instruct-RU-Spellcheck-fine-tuned)

| Metric Type | Precision (%) | Recall (%) | F1 (%) |
| ----------- | ------------- | ---------- | ------ |
| Overall     | 51.90         | 49.65      | 50.75  |
| SPELL       | 54.09         | 53.90      | 53.99  |
| CASE        | 60.00         | 50.00      | 54.55  |
| PUNCT       | 53.37         | 52.10      | 52.73  |




### SAGE-FREDT5-Large (SberDevices)


| Metric Type | Precision (%) | Recall (%) | F1 (%) |
| ----------- | ------------- | ---------- | ------ |
| Overall     | 64.42         | 73.73      | 68.76  |
| SPELL       | 59.91         | 65.42      | 62.55  |
| CASE        | 17.24         | 47.62      | 25.32  |
| PUNCT       | 11.38         | 25.75      | 15.78  |




### Model Comparison

Comparison of SAGE metrics for different models trained on different datasets. (D) — only on RU_SPELLCHECK_DEVICE, (M) — mixed data, (B) — only on SPELLCHECK_BENCHMARK**


| Metric Type | Metric    | Qwen2.5-7B(M) | Qwen2.5-14B(M) | Ministral-8B(D) | Ministral-8B(M) | Ministral-8B(B) | sage-fredt5-large |
| ----------- | --------- | ------------- | -------------- | --------------- | --------------- | --------------- | ----------------- |
| Overall     | Precision | 50.16         | 51.90          | 56.90           | 49.85           | 47.81           | **64.42**         |
|             | Recall    | 42.64         | 49.65          | 32.87           | 47.18           | 36.04           | **73.73**         |
|             | F1        | 46.10         | 50.75          | 41.67           | 48.48           | 41.10           | **68.76**         |
| SPELL       | Precision | 48.49         | 54.09          | **56.10**       | 52.32           | 46.02           | **59.91**         |
|             | Recall    | 42.96         | 53.90          | 29.45           | 52.50           | 32.95           | **65.42**         |
|             | F1        | 45.56         | 53.99          | 38.63           | 52.41           | 38.40           | **62.55**         |
| CASE        | Precision | 36.36         | **60.00**      | 57.14           | 54.76           | 22.22           | 17.24             |
|             | Recall    | 47.62         | 50.00          | 9.52            | **54.76**       | 4.76            | 47.62             |
|             | F1        | 41.24         | 54.55          | 16.33           | **54.76**       | 7.84            | 25.32             |
| PUNCT       | Precision | 1.61          | **53.37**      | 32.50           | 50.88           | 3.87            | 11.38             |
|             | Recall    | 38.32         | **52.10**      | 7.78            | **52.10**       | 6.59            | 25.75             |
|             | F1        | 43.99         | **52.73**      | 12.56           | 51.48           | 4.88            | 15.78             |


We can see that my models get better results in CASE and PUNCTUACTION than sage-fredt5-large.

## Honorable Mention

New dataset give interesting opportunitues in text classification task, because
we can try to predict the device that the person uses. This maybe useful for
improving spellcheck because typing on different device causes different error.
Also in maybe useful in Criminalistics. Here are results of simple test. I’ve
fine-tuned DeepPavlov/rubert-base-cased for predicting device label using typed
text.


| Metric                  | Value   |
| ----------------------- | ------- |
| eval_loss               | 0.73699 |
| eval_accuracy           | 0.68919 |
| eval_f1                 | 0.68411 |
| eval_precision          | 0.69390 |
| eval_recall             | 0.68919 |
| epoch                   | 5.0     |




## Citation

```bibtex
@misc{viktorov2026spellcheck,
  author = {Boris Viktorov},
  title = {Supervised Fine-Tuning of Decoder-Only Model for Russian Spellcheck Task},
  year = {2026}
}
```

