# Validation record

Performed in the development environment on 2026-10-12:

- Python 3.12, CPU PyTorch 2.6.0, NumPy 2.3.5.
- Transformers 4.57.1, PEFT 0.17.1, Accelerate 1.10.1.
- Seven core tests: analytic continuous-time likelihood, censoring gradients,
  administrative horizon, monotone curves, training-only preprocessing, duplicate-ID
  rejection, and 3D-CNN gradient/checkpoint round-trip.
- One Gemma interface test: a tiny randomly initialized Gemma3 with LoRA, actual
  continuous-prefix insertion, survival backward, evidence-language backward and
  short generation. No pretrained weights are downloaded for this test.
- End-to-end synthetic CPU experiment: 120 patients; 72 train, 24 validation, 24 test;
  12 epochs using `configs/smoke.json` except the generator count is 120.
- Checkpoint reload, curve plotting and clinical-input sensitivity inference completed.
- Smoke held-out NLL: 4.94791784; concordance at final horizon: 0.72064777.
  These small synthetic results are software checks, not clinical evidence or a claim
  of superiority. Synthetic signals intentionally depend on both modalities.

Not validated here:

- Downloading/loading the gated MedGemma 1.5 4B checkpoint.
- CUDA/BF16 runtime, memory fit, bitsandbytes 4-bit training, multi-GPU behavior.
- Real CT preprocessing/clinical validity, pretrained 3D token alignment.
- Clinical correctness, explanation faithfulness, external survival calibration.

The repository is a research starting point. The optional scikit-survival evaluator
and NIfTI converter have separate dependencies and should be exercised on your server.
No patient records, generated synthetic datasets, learned weights or authentication
tokens are included in the source archive.
