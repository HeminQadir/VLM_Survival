"""Generate artificial volumes and continuous right-censored event times."""
import argparse
import csv
from pathlib import Path
import numpy as np


def generate(output, patients=240, seed=7, shape=24):
    if patients < 30:
        raise ValueError('Use at least 30 patients.')
    out = Path(output)
    if (out/'manifest.csv').exists():
        raise FileExistsError('Choose a new output directory; refusing to replace a dataset.')
    (out/'volumes').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(patients)
    split = {int(j): ('train' if i < .6*patients else 'validation' if i < .8*patients else 'test') for i, j in enumerate(order)}
    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, shape)]*3, indexing='ij'))
    rows = []
    for i in range(patients):
        visual = rng.normal()
        age = rng.uniform(35, 85)
        albumin = rng.uniform(24, 48)
        bilirubin = rng.uniform(5, 60)
        afp = np.exp(rng.normal(4, 1))
        ecog = int(rng.integers(0, 3))
        # A soft sphere in noise, not medically realistic liver cancer or CT.
        sphere = np.exp(-(grid**2).sum(0)/(0.12+0.025*np.tanh(visual)))
        volume = (rng.normal(0, .025, sphere.shape) + (visual+2)*sphere).astype('float32')
        pid = f'synth_{i:05d}'
        np.save(out/'volumes'/f'{pid}.npy', volume)
        # Arbitrary associations; no claim of calibrated HCC biology.
        risk = .8*visual - .055*(albumin-36) + .22*ecog
        death = rng.exponential(650/np.exp(risk))
        followup = rng.uniform(200, 1500)
        row = dict(patient_id=pid, ct_path=f'volumes/{pid}.npy', time_days=max(.01, min(death, followup)),
                   event=int(death <= followup), split=split[i], age=round(age, 2),
                   albumin_g_L=round(albumin, 2), bilirubin_umol_L=round(bilirubin, 2),
                   afp_ng_mL=round(afp, 2), ecog=ecog)
        if rng.random() < .1:
            row['afp_ng_mL'] = ''
        rows.append(row)
    with (out/'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {patients} synthetic patients to {out}/manifest.csv')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/synthetic')
    p.add_argument('--patients', type=int, default=240)
    p.add_argument('--shape', type=int, default=24)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    generate(a.output, a.patients, a.seed, a.shape)
