"""Predict curves, evaluate outcomes, and report auditable input sensitivities."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import read_manifest, PatientDataset, collate_patients
from .models import build_model
from .survival import survival_curves, survival_nll
from .train import move_batch, load_trainable


def concordance(times, events, risk):
    concordant, comparable = 0., 0
    for i in range(len(times)):
        if events[i] != 1:
            continue
        later = times > times[i]
        comparable += int(later.sum())
        concordant += float((risk[i] > risk[later]).sum()+.5*(risk[i] == risk[later]).sum())
    return concordant/comparable if comparable else None


@torch.no_grad()
def run(checkpoint, manifest=None, split='test', explanations=3, verbalize=False, device=None):
    root = Path(checkpoint)
    cfg = json.loads((root/'config.json').read_text())
    prep = json.loads((root/'preprocessor.json').read_text())
    if device:
        cfg['device'] = device
    path = manifest or cfg['manifest']
    # For new patients (--split all), outcomes may be completely absent.
    labeled = split != 'all'
    rows = read_manifest(path, cfg['features'], require_splits=labeled, require_outcomes=labeled)
    rows = rows if split == 'all' else [r for r in rows if r['split'] == split]
    if not rows:
        raise ValueError('No patients selected.')
    dataset = PatientDataset(rows, path, prep, cfg)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_patients)
    model = build_model(cfg)
    load_trainable(model, root/'best.pt')
    model.eval()
    curves, losses = [], []
    for raw in loader:
        batch = move_batch(raw, cfg['device'])
        logits = model(batch)
        curves.extend(survival_curves(logits).cpu().tolist())
        if labeled:
            losses.extend(survival_nll(logits, batch['time'], batch['event'], cfg['time_edges'], 'none').cpu().tolist())
    curves = np.asarray(curves)
    output = root/f'predictions_{split}.json'
    payload = {'time_days': cfg['time_edges'][1:],
               'patients': [{'patient_id': r['patient_id'], 'survival': s.tolist()} for r, s in zip(rows, curves)]}
    if labeled:
        horizon = cfg['time_edges'][-1]
        original_time = np.array([float(r['time_days']) for r in rows])
        times = np.minimum(original_time, horizon)
        events = np.array([int(float(r['event'])) for r in rows])*(original_time <= horizon)
        payload['metrics'] = {'mean_nll': float(np.mean(losses)),
            'harrell_concordance_at_final_horizon': concordance(times, events, 1-curves[:, -1]),
            'note': 'Not a calibration metric; use evaluate.py for IPCW Brier/calibration diagnostics.'}
        print(json.dumps(payload['metrics'], indent=2))
    output.write_text(json.dumps(payload, indent=2))
    reports = []
    for index in range(min(explanations, len(rows))):
        row = rows[index]
        original = curves[index]
        sensitivities = []
        if cfg.get('use_tabular', True):
            for j, feature in enumerate(cfg['features']):
                altered = dict(row)
                altered[feature] = str(prep['median'][j])
                changed = PatientDataset([altered], path, prep, cfg)
                batch = move_batch(collate_patients([changed[0]]), cfg['device'])
                reference = survival_curves(model(batch))[0].cpu().numpy()
                sensitivities.append({'feature': feature, 'recorded_value': row[feature],
                    'replacement_value': prep['median'][j],
                    'original_minus_replacement_survival': (original-reference).tolist()})
        report = {'patient_id': row['patient_id'], 'time_days': cfg['time_edges'][1:],
            'predicted_survival': original.tolist(), 'feature_sensitivities': sensitivities,
            'interpretation': 'One input replaced at a time by its training median. Negative differences mean lower predicted survival than that replacement. Not causal effects or additive attributions. Correlated features can make replacements implausible.',
            'image_claims': 'No supervised finding labels. This model cannot verify named CT abnormalities.',
            'clinical_correctness': 'Not established; independent clinical evaluation required.'}
        if verbalize:
            if cfg['backend'] != 'medgemma':
                raise ValueError('--verbalize requires a MedGemma checkpoint.')
            # Compact, real model-computed facts only. No invented guideline statements.
            compact = {'horizon_days': cfg['time_edges'][-1], 'survival': round(float(original[-1]), 3),
                'sensitivities_at_horizon': {s['feature']: round(s['original_minus_replacement_survival'][-1], 3) for s in sensitivities}}
            batch = move_batch(collate_patients([dataset[index]]), cfg['device'])
            report['unverified_generated_text'] = model.verbalize(batch, json.dumps(compact))
        reports.append(report)
    (root/f'evidence_{split}.json').write_text(json.dumps(reports, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for row, curve in zip(rows[:5], curves[:5]):
        ax.plot(cfg['time_edges'], np.r_[1, curve], label=row['patient_id'])
    ax.set(xlabel='Days since baseline', ylabel='Predicted survival probability', ylim=(0, 1),
           title='Research predictions — clinical correctness not established')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/f'curves_{split}.png', dpi=160)
    plt.close(fig)
    print(f'Wrote {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest')
    p.add_argument('--split', choices=['train', 'validation', 'test', 'all'], default='test')
    p.add_argument('--explanations', type=int, default=3)
    p.add_argument('--verbalize', action='store_true')
    p.add_argument('--device')
    a = p.parse_args()
    run(a.checkpoint, a.manifest, a.split, a.explanations, a.verbalize, a.device)
