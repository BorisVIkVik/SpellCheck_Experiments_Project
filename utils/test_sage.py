import os
import torch
from sage.utils import DatasetsAvailable, load_available_dataset_from_hf
from sage.spelling_correction import AvailableCorrectors
from sage.spelling_correction import T5ModelForSpellingCorruption

# corrector_fred_95m = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_fredt5_distilled_95m.value)
corrector_mt5 = T5ModelForSpellingCorruption.from_pretrained(AvailableCorrectors.sage_mt5_large.value)
print('Downloaded')
# corrector_fred_95m.model.to(torch.device("cuda:0"))
# corrector_mt5.model.to(torch.device("cuda:0"))
print('Cuda')
# metrics = corrector_fred_95m.evaluate("RUSpellRU", metrics=["errant", "ruspelleval"], batch_size=32)
# print(metrics)
# {'CASE_Precision': 94.41, 'CASE_Recall': 92.55, 'CASE_F1': 93.47, 'SPELL_Precision': 77.52, 'SPELL_Recall': 64.09, 'SPELL_F1': 70.17, 'PUNCT_Precision': 86.77, 'PUNCT_Recall': 80.59, 'PUNCT_F1': 83.56, 'YO_Precision': 46.21, 'YO_Recall': 73.83, 'YO_F1': 56.84, 'Precision': 83.48, 'Recall': 74.75, 'F1': 78.87}
# print(
# sources, corrections = load_available_dataset_from_hf(DatasetsAvailable.RUSpellRU.name, for_labeler=True, split="train")
# print(len(sources), len(corrections))

metrics = corrector_mt5.evaluate("RUSpellRU", metrics=["ruspelleval"], batch_size=16)
print(metrics)
# {'Precision': 75.94, 'Recall': 88.15, 'F1': 81.59}
