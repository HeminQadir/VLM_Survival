# Data contract

## Endpoint and patient selection

Specify the subtype: hepatocellular carcinoma (HCC), intrahepatic cholangiocarcinoma,
and other liver malignancies are not interchangeable. The default disease string
does not assume HCC. Define one baseline and one endpoint before fitting anything.

For overall survival:

- `time_days`: elapsed days from baseline CT/prediction date to death OR last known alive.
- `event=1`: death was observed at that time.
- `event=0`: right-censored at last confirmed alive follow-up.
- Positive finite times only. Zero-day outcomes need a prespecified clinical policy;
  do not add arbitrary days to real records just to satisfy validation.
- Include survivors. Do not train only on deaths and interpret the result as cohort survival.
- Inputs must have been available at baseline. Future treatments, response measures,
  later-stage assessments or subsequent lab results cause leakage.

This implementation assumes right censoring that is independent of event time given
the modeled information. Informative dropout, delayed entry/left truncation, recurrent
events and competing risks need additional methods. All-cause death does not identify
whether cancer or liver failure caused death. It does not identify treatment benefit.

## CSV schema

```csv
patient_id,ct_path,time_days,event,split,age,albumin_g_L,bilirubin_umol_L,afp_ng_mL,ecog
example_001,volumes/example_001.npy,410,1,train,65,34,22,120,1
example_002,volumes/example_002.npy,700,0,validation,59,40,15,,0
example_003,volumes/example_003.npy,500,0,test,70,36,20,80,1
```

These are fabricated examples, not enough records to train. Features in the config
must appear as columns. Currently numeric features only are supported; encode nominal
categories explicitly as separate indicator columns (do not imply an ordinal scale).
Missing values can be empty or `NA`. Units are encoded in field names and must be
consistent. Do not place identifiers, dates, outcomes or post-baseline fields in features.
Reserve test patients before model design; use institutional or temporal splits when
appropriate. Current validation requires one row per patient with a unique ID.

`ct_path` is relative to the manifest directory, or an absolute local path. Keep
the CSV and CTs out of GitHub. Model outputs also include patient IDs; protect them.

## CT volume route

1. Convert an authorized, de-identified DICOM series to NIfTI with a trusted local tool.
2. Confirm it represents a single appropriate series and that values are Hounsfield units.
3. Choose a reproducible phase. Do not mix arterial and portal-venous scans silently.
4. Convert with:

```bash
python -m vlm_survival.prepare_ct --input /secure/patient.nii.gz \
  --output data/real/volumes/patient.npy --shape 64 96 96
```

The utility canonicalizes orientation, windows HU and resizes the whole volume. It
records geometry in a sidecar. This is an explicit simple baseline: it can distort
physical scale and remove small lesions. It does not segment the liver, identify
lesions or register multiple phases. Validate preprocessing visually on your data.
For research, consider a liver ROI with documented spacing and a validated pretrained
encoder. Keep all required anatomy for the disease question; do not silently crop it.

Set `image_shape` to `[64,96,96]`, `image_mode` to `volume`, and adjust batch size.
The model currently accepts one channel/phase. A multiphase model needs explicit code
changes and alignment checks. Missing phases should not be silently copied.

## Cached pretrained 3D encoder route

This is usually preferable to fitting the supplied small CNN on limited survival data.
Export each patient's spatial features as a finite float32 array `[N,D]`:

```python
np.save('embeddings/patient.npy', tokens.astype('float32'))
```

Add an `embedding_path` column, set `image_mode` to `tokens` and `feature_dim` to `D`.
`D` must be divisible by four for the adapter attention. Use a fixed `N` per batch,
or batch size 1. Keep token normalization consistent and record the encoder revision.

The cached features must already contain spatial positional information. A set
of unordered tokens alone cannot preserve anatomical coordinates in this adapter.
Precompute with a frozen encoder and patient-safe splits; train any dataset-specific
normalization or encoder updates without test patients. This route freezes the encoder
by construction. There is no hidden download of a particular 3D foundation model.

## Follow-up intervals

`time_edges` starts at 0 and increases in days. The model assumes a constant hazard
within each interval. It uses the exact fraction of an interval observed before
censoring. Deaths after the last edge become administratively censored at that edge
for the training objective. Report this horizon. Predictions beyond it are not provided.
Choose edges using clinical utility and training/validation support, never test outcomes.
