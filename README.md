# VLM Survival

Readable research code for personalized **all-cause survival curves** from 3D CT
and baseline tabular data, with a **MedGemma 1.5 4B + LoRA** backend and a small CPU
debugging backend. Designed as a starting point for a CVPR 2027 research project.

## What is implemented

- Synthetic 3D volumes, clinical tables, continuous death times and censoring.
- A small trainable 3D CNN, or loading cached spatial tokens from your own encoder.
- A learned visual-token adapter and a numeric clinical token.
- Clinical text with named fields and explicit missingness.
- Actual MedGemma language-decoder integration and LoRA training.
- A continuous-time piecewise-exponential survival head. Censoring inside an
  interval is handled correctly; curves are monotone by construction.
- An optional input-derived table transcription loss, using **no manual rationale labels**.
- Patient-level split checks, training-only imputation/scaling, validation model selection.
- Numerical sensitivity reports, curve plots and optional MedGemma verbalization.
- Optional IPCW Brier scores and concordance through scikit-survival.

**What this does not establish:** clinically correct CT interpretation or medical
reasoning from survival labels alone. The evidence task checks table transcription;
it is not a clinical explanation target. Generated prose is explicitly marked
unverified. The small CNN is randomly initialized, not a medical foundation encoder.
See [research scope](docs/RESEARCH.md) before interpreting results.

## 1. Install on your server

Python 3.10–3.12 is recommended. Start from the repository root in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch 2.6 wheel appropriate for your GPU/driver. For a compatible CUDA
12.4 setup, for example:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[medgemma,ct]'
```

CPU-only smoke tests:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

This code targets **Transformers 4.57.1 and PEFT 0.17.1**. Do not initially upgrade
Transformers to a different major version. No torchvision or flash-attention is
required. A clean environment avoids unrelated torchvision/librosa import failures.
The exact resolved foundation-model revision is saved after loading.

## 2. First run: no model download or GPU

```bash
python -m unittest discover -s tests -p test_core.py -v
python -m vlm_survival.synthetic --output data/synthetic --patients 240
python -m vlm_survival.train --config configs/smoke.json
python -m vlm_survival.predict --checkpoint runs/smoke
```

Outputs include `best.pt`, `config.json`, `preprocessor.json`, `environment.json`,
`history.json`, `predictions_test.json`, `evidence_test.json` and `curves_test.png`.
Synthetic volumes are smooth blobs in noise, **not realistic liver CTs**. Outcomes
are generated with arbitrary coefficients, not a clinical liver-cancer risk formula.

The generator refuses to overwrite an existing manifest; training refuses to
overwrite a checkpoint. Use new paths when rerunning. All paths in configs are
relative to your current working directory, so run commands from the repository root.

## 3. Run the MedGemma backend

First accept the model's terms at:
https://huggingface.co/google/medgemma-1.5-4b-it

Authenticate locally; do not put tokens in code or configs:

```bash
hf auth login
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
python -m vlm_survival.train --config configs/medgemma.json
python -m vlm_survival.predict --checkpoint runs/medgemma
```

The Gemma interface test uses a tiny, randomly initialized Gemma3 model and does not
download MedGemma. It exercises the actual decoder interface, LoRA gradients and
the separate evidence task. It is not a test of pretrained medical performance.

The default MedGemma config uses your generated synthetic data to check GPU training.
Start there before using real CTs. Unquantized weights alone need roughly 8 GB at
BF16; total training memory is larger. Begin with batch size 1 and 8 visual tokens.
A 24 GB GPU may be adequate for short inputs with LoRA, but memory depends on hardware
and inputs. Non-BF16 devices use FP32 in this implementation and need more memory.

Optional 4-bit base weights:

```bash
python -m pip install -e '.[quantization]'
```

Set `"load_in_4bit": true` in a copied config. This path is provided for your server
and is not GPU-tested here. Do not run `.to(device)` on the entire quantized wrapper;
the code places new trainable modules separately. Start with one GPU; distributed
training and automatic resume are intentionally not implemented in this version.

## 4. How training works

The 3D adapter produces continuous prefix tokens. A separate numeric token encodes
standardized clinical values and missingness. These are inserted after the first
chat token, followed by clinical text. The final input hidden state predicts
interval-specific integrated hazards. `exp(-cumulative hazard)` gives survival.

The original MedGemma 2D tower is **bypassed**. Its pretrained visual grounding does
not automatically transfer to this new adapter. Learning that alignment is a research
problem, especially without reports. The included CNN is useful for debugging, not a
credible pretrained 3D baseline by itself.

There are two separate forward/backward passes when `evidence_weight > 0`:

1. **Survival:** only baseline inputs are visible; follow-up and death flags are loss targets.
2. **Evidence:** ask the model to repeat supplied clinical fields and train on that
   answer. No event labels or outcome-generated explanations are supplied.

This separation prevents a teacher-forced answer from leaking into the survival head.
The second task is deliberately modest and auditable. It does not prove clinical
reasoning. The current code does not implement teacher distillation, clinical concept
labels, guideline retrieval, RL or visual pseudo-labeling.

## 5. Prepare your own data

Read [DATA.md](docs/DATA.md) for the manifest, preprocessing and cached-encoder route.
Each patient needs baseline inputs, elapsed follow-up time and an event flag. You
must include patients alive at last follow-up, not only patients who died.

For fresh patients without outcomes:

```bash
python -m vlm_survival.predict --checkpoint runs/medgemma \
  --manifest data/new_patients/manifest.csv --split all
```

This manifest can omit `event`, `time_days` and `split`. It must retain the same
clinical feature columns and image input format used at training time.

## 6. Explanations and evaluation

`evidence_test.json` computes changes in the survival curve when one clinical field
is replaced by its training median. Both the numerical token and textual field are
updated together. These are **model sensitivities**, not causal effects, additive
attributions, or guaranteed plausible alternate patients. Correlated fields can make
these replacements unrealistic. Missing-to-median comparisons also change missingness.

Optional free-text output:

```bash
python -m vlm_survival.predict --checkpoint runs/medgemma --explanations 3 --verbalize
```

The generated text is saved under `unverified_generated_text`. The source numeric
record is always retained. Review generated numbers and claims; there is no automatic
clinical correctness verifier. `max_prompt_tokens` may need increasing for this task.

Additional survival metrics:

```bash
python -m pip install -e '.[evaluation]'
python -m vlm_survival.evaluate --checkpoint runs/medgemma
```

IPCW evaluation can reject a dataset with insufficient training follow-up support.
Do not fix that by arbitrarily deleting inconvenient test patients. Define evaluation
horizons and administrative censoring prospectively. Confidence intervals, external
cohorts and subgroup calibration still need to be added for a paper.

## 7. Make changes in small steps

| Goal | File / setting |
|---|---|
| Change clinical variables or disease | Config `features`, `disease` |
| Change follow-up horizon | Config `time_edges`, set without test-set tuning |
| Replace the 3D backbone | `models.py: VolumeEncoder` or cached `image_mode=tokens` |
| Change adapter/fusion | `models.py: ImageAdapter`, `MedGemmaSurvival.inputs` |
| Change likelihood | `survival.py` and its analytic tests |
| Change the checkable language task | `models.py: evidence_loss` |
| Control trainable LLM weights | LoRA settings in `models.py` / config |
| Table-only baseline | `use_image=false`, unique output directory |
| Image-only baseline | `use_tabular=false`, `evidence_weight=0`, unique output |
| Remove auxiliary supervision | `evidence_weight=0` |

## 8. Troubleshooting

- **401/403 from Hugging Face:** accept terms and authenticate with an account that
  has access. No model weights or credentials are distributed here.
- **CUDA out of memory:** reduce prompt length, visual-token count and evidence answer
  length; keep batch size 1; try optional 4-bit loading. Reducing accumulation alone
  does not reduce per-microbatch activation memory.
- **Prompt exceeds max_prompt_tokens:** increase the limit or explicitly select fewer
  fields; the code deliberately refuses silent clinical-text truncation.
- **Manifest/image shape error:** check relative paths, array shape, phase and config.
- **Checkpoint key mismatch:** use the saved config and pinned dependency versions.
- **Clinical explanation sounds convincing but unsupported:** treat it as a failure,
  retain the numeric evidence record, and evaluate independently.

Please provide the complete traceback, command, config (without patient records or
secrets), Python/PyTorch/Transformers versions, GPU model and VRAM when reporting errors.

## Clone this repository

```bash
git clone https://github.com/HeminQadir/VLM_Survival.git
cd VLM_Survival
```

Then follow the installation and smoke-test instructions above. Only source and
documentation belong in git. `data/`, `runs/`, weights and patient images are ignored.
MedGemma weights have their own terms; no model license is conferred by this source
package. Validation status is recorded in `docs/VALIDATION.md`.
