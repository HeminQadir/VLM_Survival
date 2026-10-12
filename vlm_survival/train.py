"""Single-device training with explicit patient splits and validation checkpoints."""
import argparse
import json
import random
import platform
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import read_manifest, fit_preprocessor, PatientDataset, collate_patients
from .models import build_model
from .survival import survival_nll


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_trainable(model, path):
    # Save adapters/CNN/head only, never copy gigabytes of frozen foundation weights.
    state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    torch.save(state, path)


def load_trainable(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError('Trainable parameter names differ. Use the saved config and pinned dependencies.')
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f'Unexpected checkpoint keys: {result.unexpected_keys}')


@torch.no_grad()
def validate(model, loader, cfg):
    model.eval()
    total, count = 0., 0
    for raw in loader:
        batch = move_batch(raw, cfg['device'])
        loss = survival_nll(model(batch), batch['time'], batch['event'], cfg['time_edges'], reduction='none')
        total += loss.sum().item()
        count += len(loss)
    return total/count


def train(config_path):
    cfg = json.loads(Path(config_path).read_text())
    seed = cfg.get('seed', 17)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    out = Path(cfg['output'])
    if (out/'best.pt').exists():
        raise FileExistsError('Checkpoint exists. Choose a new output path; automatic resume is not implemented.')
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(cfg['manifest'], cfg['features'])
    training = [r for r in rows if r['split'] == 'train']
    if sum(int(float(r['event'])) == 1 and float(r['time_days']) <= cfg['time_edges'][-1] for r in training) == 0:
        raise ValueError('No training events within prediction horizon.')
    prep = fit_preprocessor(training, cfg['features'])
    loaders = {}
    for split in ['train', 'validation']:
        subset = [r for r in rows if r['split'] == split]
        loaders[split] = DataLoader(PatientDataset(subset, cfg['manifest'], prep, cfg),
            batch_size=cfg['batch_size'], shuffle=split == 'train', num_workers=0, collate_fn=collate_patients)
    model = build_model(cfg)
    (out/'config.json').write_text(json.dumps(cfg, indent=2))
    (out/'preprocessor.json').write_text(json.dumps(prep, indent=2))
    environment = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                   'device': cfg['device'], 'seed': seed, 'patients': {s: sum(r['split'] == s for r in rows) for s in ['train', 'validation', 'test']},
                   'training_ids': [r['patient_id'] for r in training]}
    if cfg['backend'] == 'medgemma':
        import transformers, peft
        environment.update(transformers=transformers.__version__, peft=peft.__version__,
                           resolved_model_revision=getattr(model.base.config, '_commit_hash', None))
        # Reload the exact model revision for prediction instead of a moving main.
        if environment['resolved_model_revision']:
            cfg['model_revision'] = environment['resolved_model_revision']
            (out/'config.json').write_text(json.dumps(cfg, indent=2))
    (out/'environment.json').write_text(json.dumps(environment, indent=2))
    parameters = [p for p in model.parameters() if p.requires_grad]
    print(f'Trainable parameters: {sum(p.numel() for p in parameters):,}', flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    accumulation = cfg.get('accumulation_steps', 1)
    if accumulation < 1:
        raise ValueError('accumulation_steps must be positive.')
    best = float('inf')
    history = []
    for epoch in range(cfg['epochs']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, n = 0., 0
        for step, raw in enumerate(loaders['train']):
            batch = move_batch(raw, cfg['device'])
            logits = model(batch)
            survival = survival_nll(logits, batch['time'], batch['event'], cfg['time_edges'])
            # Scale by the true number of microbatches in the final accumulation group.
            group_start = (step//accumulation)*accumulation
            group_size = min(accumulation, len(loaders['train'])-group_start)
            (survival/group_size).backward()
            evidence = 0.
            if cfg.get('evidence_weight', 0) > 0:
                aux = model.evidence_loss(batch)
                (cfg['evidence_weight']*aux/group_size).backward()
                evidence = aux.detach().item()
            if (step+1) % accumulation == 0 or step+1 == len(loaders['train']):
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += survival.detach().item()*len(batch['time'])
            n += len(batch['time'])
            if not np.isfinite(total):
                raise FloatingPointError('Nonfinite loss. Inspect inputs, dtype and learning rate.')
        val = validate(model, loaders['validation'], cfg)
        record = {'epoch': epoch+1, 'train_survival_nll': total/n, 'validation_survival_nll': val,
                  'last_batch_evidence_ce': evidence}
        history.append(record)
        print(json.dumps(record), flush=True)
        (out/'history.json').write_text(json.dumps(history, indent=2))
        if val < best:
            best = val
            save_trainable(model, out/'best.pt')
    print(f'Best checkpoint: {out}/best.pt', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    train(parser.parse_args().config)
