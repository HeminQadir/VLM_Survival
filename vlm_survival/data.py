"""Manifest validation, training-only preprocessing, and cached image loading."""
import csv
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

RESERVED = {'patient_id', 'ct_path', 'embedding_path', 'time_days', 'event', 'split', 'site_id'}


def number(value):
    if value is None or str(value).strip().lower() in ('', 'na', 'nan', 'none', 'null'):
        return float('nan')
    out = float(value)
    if not math.isfinite(out):
        raise ValueError('Infinite numeric clinical value.')
    return out


def read_manifest(path, features, require_splits=True, require_outcomes=True):
    if set(features) & RESERVED:
        raise ValueError('Outcome, ID, path and split fields cannot be model features.')
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        required = {'patient_id', *features}
        if require_outcomes:
            required |= {'time_days', 'event'}
        if require_splits:
            required.add('split')
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing manifest columns: {sorted(missing)}')
        rows = list(reader)
    ids = [r['patient_id'] for r in rows]
    if not rows or any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Manifest must contain unique, nonempty patient IDs.')
    for row in rows:
        if require_outcomes:
            t, e = number(row['time_days']), number(row['event'])
            if not math.isfinite(t) or t <= 0 or e not in (0, 1):
                raise ValueError(f'Invalid outcome for {row["patient_id"]}')
        if require_splits and row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('split must be train, validation or test.')
        for feature in features:
            number(row[feature])
    if require_splits and set(r['split'] for r in rows) != {'train', 'validation', 'test'}:
        raise ValueError('All three predefined patient splits are required.')
    return rows


def fit_preprocessor(rows, features):
    values = np.array([[number(r[f]) for f in features] for r in rows], dtype=float)
    if np.any(np.all(np.isnan(values), axis=0)):
        raise ValueError('A feature is entirely missing in training; remove it explicitly.')
    median = np.nanmedian(values, axis=0)
    filled = np.where(np.isnan(values), median, values)
    return {'features': features, 'median': median.tolist(),
            'mean': filled.mean(0).tolist(), 'scale': np.maximum(filled.std(0), 1e-6).tolist()}


def transform_row(row, prep):
    values = np.array([number(row[f]) for f in prep['features']])
    missing = np.isnan(values)
    values = np.where(missing, prep['median'], values)
    values = (values - np.asarray(prep['mean'])) / np.asarray(prep['scale'])
    return np.concatenate([values, missing.astype(float)]).astype('float32')


def clinical_text(row, features):
    # Explicit whitelist excludes outcomes and IDs. No arbitrary row serialization.
    return '\n'.join(f'{f}: {row[f] if math.isfinite(number(row[f])) else "not measured"}' for f in features)


class PatientDataset(Dataset):
    def __init__(self, rows, manifest, prep, config):
        self.rows, self.root, self.prep, self.config = rows, Path(manifest).resolve().parent, prep, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, cfg = self.rows[index], self.config
        mode = cfg['image_mode']
        if cfg.get('use_image', True):
            column = 'embedding_path' if mode == 'tokens' else 'ct_path'
            if not row.get(column):
                raise ValueError(f'{row["patient_id"]}: missing {column}')
            path = Path(row[column])
            array = np.load(path if path.is_absolute() else self.root/path, allow_pickle=False)
            if mode == 'volume':
                if array.shape != tuple(cfg['image_shape']):
                    raise ValueError(f'{path}: expected volume shape {cfg["image_shape"]}, got {array.shape}')
                array = array[None]
            elif array.ndim != 2 or array.shape[-1] != cfg['feature_dim']:
                raise ValueError('Cached tokens must be [N, feature_dim].')
            if not np.isfinite(array).all():
                raise ValueError(f'Nonfinite image input: {path}')
        else:
            array = np.zeros((1, *cfg['image_shape']) if mode == 'volume' else (1, cfg['feature_dim']), dtype='float32')
        tab = transform_row(row, self.prep)
        text = clinical_text(row, self.prep['features']) if cfg.get('use_tabular', True) else 'Clinical data withheld for image-only ablation.'
        if not cfg.get('use_tabular', True):
            tab[:] = 0
        return {'image': torch.from_numpy(array.astype('float32')), 'tabular': torch.from_numpy(tab),
                'clinical_text': text, 'patient_id': row['patient_id'],
                'time': torch.tensor(float(row.get('time_days') or 1)),
                'event': torch.tensor(float(row.get('event') or 0))}


def collate_patients(items):
    images = [x['image'] for x in items]
    if len({tuple(x.shape) for x in images}) != 1:
        raise ValueError('Use a fixed token count per cached image or batch_size=1.')
    batch = {k: torch.stack([x[k] for x in items]) for k in ['image', 'tabular', 'time', 'event']}
    batch.update({k: [x[k] for x in items] for k in ['clinical_text', 'patient_id']})
    return batch
