"""Paired, fresh-start experiments. Existing output directories are never reused."""
import argparse
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from model import GPT

ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ('vocab_size', 'd_model', 'n_heads', 'd_ff', 'n_layers', 'use_rope', 'activation')
BASE = dict(vocab_size=513, d_model=384, n_heads=6, d_ff=1536, n_layers=8,
            use_rope=True, activation='relu', batch_size=4, context_length=128,
            learning_rate=0.0003, weight_decay=0.01)
VARIANTS = {'baseline_control': {}, 'rope_off': {'use_rope': False},
            'gelu': {'activation': 'gelu'}, 'no_weight_decay': {'weight_decay': 0.0}}

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def batch(ids, c, rng, device):
    starts = torch.randint(len(ids) - c['context_length'], (c['batch_size'],), generator=rng).tolist()
    x = torch.stack([ids[i:i+c['context_length']] for i in starts])
    y = torch.stack([ids[i+1:i+c['context_length']+1] for i in starts])
    return x.to(device), y.to(device)

def loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())

def run(args, variant, data, hashes, device):
    c = dict(BASE, **VARIANTS[variant], seed=args.seed, max_steps=args.steps,
             eval_interval=args.eval_interval, eval_batches=args.eval_batches)
    if args.tiny:
        c.update(d_model=32, n_heads=4, d_ff=128, n_layers=2, batch_size=2, context_length=16)
    out = args.output / f'{variant}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = GPT(**{k:c[k] for k in MODEL_KEYS}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    manifest = dict(config=c, variant=variant, device=str(device), torch=torch.__version__,
                    python=platform.python_version(), data_sha256=hashes,
                    source_sha256={f:digest(ROOT/f) for f in ['model.py', 'experiments.py']},
                    parameters=sum(p.numel() for p in model.parameters()),
                    validation_seed=args.seed+2, status='running',
                    note='Fresh start; fixed validation windows; CPU deterministic algorithms; cross-device bitwise equality is not promised.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    start = time.perf_counter()
    with (out/'loss.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'val_loss', 'elapsed_seconds'])
        for step in range(1, args.steps+1):
            model.train()
            x, y = batch(data[0], c, train_rng, device)
            optimizer.zero_grad(set_to_none=True)
            value = loss(model, x, y)
            value.backward()
            optimizer.step()
            if step == 1 or step % args.eval_interval == 0 or step == args.steps:
                model.eval()
                val_rng = torch.Generator().manual_seed(args.seed+2)
                with torch.no_grad():
                    validation = sum(loss(model, *batch(data[1], c, val_rng, device)).item() for _ in range(args.eval_batches))/args.eval_batches
                if device.type == 'mps': torch.mps.synchronize()
                row = [step, value.item(), validation, time.perf_counter()-start]
                writer.writerow(row)
                f.flush()
                print(variant, *row, flush=True)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=c, step=step,
                    torch_rng_state=torch.get_rng_state(), train_rng_state=train_rng.get_state()), out/'checkpoint.pt')
    manifest['status'] = 'completed'
    manifest['checkpoint_sha256'] = digest(out/'checkpoint.pt')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=['all', *VARIANTS], default='all')
    ap.add_argument('--data-dir', type=Path, default=ROOT/'data')
    ap.add_argument('--output', type=Path, default=ROOT/'results'/'controlled')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--eval-interval', type=int, default=250)
    ap.add_argument('--eval-batches', type=int, default=20)
    ap.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--tiny', action='store_true', help='small model for verification only')
    args = ap.parse_args()
    if min(args.steps, args.eval_interval, args.eval_batches, args.threads) < 1: ap.error('numeric settings must be positive')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device)
    paths = [args.data_dir/f for f in ['train_tokens.pt', 'validation_tokens.pt']]
    data = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    for ids in data:
        if ids.ndim != 1 or ids.dtype != torch.long or len(ids) <= BASE['context_length'] or ids.min() < 0 or ids.max() >= BASE['vocab_size']:
            raise ValueError('Invalid token tensor: expected 1D long IDs in [0, 513), length >128')
    hashes = {p.name:digest(p) for p in paths}
    for variant in (VARIANTS if args.variant == 'all' else [args.variant]):
        run(args, variant, data, hashes, device)

if __name__ == '__main__': main()
