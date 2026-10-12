"""Optional censoring-adjusted evaluation using scikit-survival.

Run after predict. Fails clearly when training censoring support is insufficient;
does not silently extrapolate censoring probabilities or discard test patients.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import read_manifest


def evaluate(checkpoint):
    from sksurv.metrics import brier_score, integrated_brier_score, concordance_index_ipcw
    from sksurv.util import Surv
    from sksurv.nonparametric import kaplan_meier_estimator
    root = Path(checkpoint)
    cfg = json.loads((root/'config.json').read_text())
    rows = read_manifest(cfg['manifest'], cfg['features'])
    train = [r for r in rows if r['split'] == 'train']
    test = [r for r in rows if r['split'] == 'test']
    prediction = json.loads((root/'predictions_test.json').read_text())
    lookup = {p['patient_id']: p['survival'] for p in prediction['patients']}
    if set(lookup) != {r['patient_id'] for r in test}:
        raise ValueError('Predictions and test manifest IDs differ.')
    def outcomes(subset):
        return Surv.from_arrays([bool(int(float(r['event']))) for r in subset], [float(r['time_days']) for r in subset])
    y_train, y_test = outcomes(train), outcomes(test)
    times = np.asarray(prediction['time_days'], dtype=float)
    curves = np.asarray([lookup[r['patient_id']] for r in test])
    # Evaluation horizons must lie inside observed test support and model horizon.
    keep = (times >= y_test['time'].min()) & (times < y_test['time'].max()) & (times < y_train['time'].max())
    times, curves = times[keep], curves[:, keep]
    if len(times) < 2:
        raise ValueError('Need two supported horizons. Adjust predefined time_edges using training/validation only.')
    try:
        _, bs = brier_score(y_train, y_test, curves, times)
        ibs = integrated_brier_score(y_train, y_test, curves, times)
        uno = concordance_index_ipcw(y_train, y_test, 1-curves[:, -1], tau=times[-1])[0]
    except ValueError as exc:
        raise ValueError('IPCW evaluation lacks follow-up/censoring support. Define a justified administrative cutoff prospectively; do not silently trim inconvenient test patients.') from exc
    km_t, km_s = kaplan_meier_estimator(y_test['event'], y_test['time'])
    indices = np.searchsorted(km_t, times, side='right')-1
    km = np.where(indices >= 0, km_s[np.maximum(indices, 0)], 1.)
    result = {'time_days': times.tolist(), 'brier_score': bs.tolist(), 'integrated_brier_score': float(ibs),
              'uno_concordance': float(uno), 'mean_predicted_survival': curves.mean(0).tolist(),
              'cohort_km_survival': km.tolist(),
              'note': 'Mean curve versus KM is a cohort-level diagnostic, not proof of individual calibration. IPCW uses training censoring distribution; confidence intervals and subgroup calibration are not implemented.'}
    (root/'evaluation_test.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    evaluate(p.parse_args().checkpoint)
