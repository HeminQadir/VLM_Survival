# Scientific scope and CVPR 2027 plan

## Claim this implementation can support

Adapt a pretrained medical language model for CT-and-tabular survival prediction
without manual patient-level explanation labels. The auxiliary task is input-derived
table transcription, not expert clinical reasoning. Training includes outcome labels.

## Claims it cannot support by itself

- A new encoder is clinically aligned with the language decoder because dimensions match.
- A plausible generated rationale is the true reason for a predicted curve.
- All-cause death identifies cancer-specific mortality or treatment benefit.
- Strong concordance proves calibration, external validity or clinical correctness.
- This architecture alone is novel or likely to be accepted at CVPR.

## Proposed research extension

The central hypothesis is that independently checkable evidence tasks can improve
prognostic grounding without manual rationale labels. The current table-copy task is
a minimal baseline. Candidate extensions (NOT implemented) include verified numeric
comparisons, missingness reasoning, frozen-teacher visual pseudo-labels, explicitly
tracked disease references, and consistency under medically valid transformations.
Pseudo-labels and guidelines are additional supervision sources and must be disclosed.
Do not generate rationales by revealing the patient's future outcome to a teacher.

Test clinical correctness with an expert-reviewed held-out subset. That does not
contradict an annotation-free TRAINING claim. If no clinical evaluation is possible,
limit claims to survival prediction, numerical faithfulness and tested consistency.

## Minimum comparison matrix

1. Clinical-only conventional survival model.
2. 3D-image-only conventional survival model.
3. Conventional image + table fusion (the tiny model is a debugging version, not SOTA).
4. Original MedGemma slice pathway (not implemented in this adapter-first package).
5. MedGemma + 3D adapter + survival objective.
6. Same model + input-derived evidence objective.
7. Same encoder/features and data splits across comparisons where possible.

Run multiple seeds, patient-level confidence intervals, an external/temporal cohort,
calibration at supported horizons, IPCW Brier scores and clinical-only added-value
analysis. Evaluate model-generated statements separately from survival scores. Inspect
image dependence and correct handling of missing fields; a model can exploit strong
clinical covariates while ignoring its image prefix.

## HCC scope, if confirmed

HCC prognosis involves tumor extent, underlying liver dysfunction and performance
status. AASLD guidance supports this broad framework. It is not an individual causal
explanation and is not transferable unchanged to cholangiocarcinoma. No hard-coded
medical direction constraints or prognostic scores are added in this package.

## Relevant prior work

- MedGemma 1.5: https://arxiv.org/abs/2604.05081
- Medical Image Understanding Improves Survival Prediction via Visual Instruction
  Tuning: https://arxiv.org/abs/2604.18250
- ConSurv: https://proceedings.mlr.press/v298/borole25a.html
- AASLD guidance: https://pmc.ncbi.nlm.nih.gov/articles/10663390/

Swapping a language backbone is not a sufficient novelty claim. Do a focused
literature review before describing any combined method as unpublished.
