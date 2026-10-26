"""Fresh, equal-matrix-budget ReLU/GELU/SiLU/SwiGLU comparison."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import pickle
import platform
import subprocess
import time
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
from model import GPT, TokenEmbedding
from experiments import BASE, MODEL_KEYS, batch, digest, loss
from generate import generate

ROOT = Path(__file__).resolve().parent
ACTIVATIONS = ('relu', 'gelu', 'silu', 'swiglu')
MODEL_FIELDS = (*MODEL_KEYS, 'ffn_bias')
SOURCES = ('model.py', 'activation_experiments.py', 'experiments.py',
           'generate.py', 'tokenizer.py', 'summarize_ffn.py', 'test_ffn_comparison.py')

def initialize(model, seed):
    """Seed each named module separately so common modules match across FFN designs."""
    for name, module in model.named_modules():
        key = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little') % (2**63)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(key)
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
            elif isinstance(module, TokenEmbedding):
                with torch.no_grad():
                    module.weight.normal_(0, 0.02)

def fingerprint_parameters(model, common_only=False):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if common_only and '.ffn.' in name:
            continue
        h.update(name.encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def evaluate(model, ids, config, seed):
    rng = torch.Generator().manual_seed(seed)
    model.eval()
    with torch.no_grad():
        return sum(loss(model, *batch(ids, config, rng, 'cpu')).item()
                   for _ in range(config['eval_batches'])) / config['eval_batches']

def write_samples(model, config, data_dir, output):
    from tokenizer import encode, build_vocab
    with (data_dir/'bpe_merges.pkl').open('rb') as f:
        merges = pickle.load(f)
    vocab = build_vocab(merges)
    eos = 256 + len(merges)
    if eos + 1 != config['vocab_size']:
        raise ValueError('Tokenizer vocabulary mismatch')
    records = []
    for prompt in ('Once upon a time', 'The little girl found', 'One day, a rabbit'):
        torch.manual_seed(2026)
        ids = torch.tensor([encode(prompt, merges)], dtype=torch.long)
        generated = generate(model, ids, 100, config['context_length'], eos)[0].tolist()
        if generated[-1] == eos:
            generated.pop()
        text = b''.join(vocab[i] for i in generated).decode('utf-8', errors='replace')
        records.append(dict(prompt=prompt, seed=2026, max_new_tokens=100, text=text))
    (output/'samples.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))

def run(args, activation, data, data_hashes, source_hashes, suite):
    config = dict(BASE, activation=activation, ffn_bias=False, seed=args.seed,
                  max_steps=args.steps, eval_interval=args.eval_interval, eval_batches=args.eval_batches)
    output = suite/f'{activation}_seed{args.seed}'
    output.mkdir()
    torch.manual_seed(args.seed)
    model = GPT(**{key: config[key] for key in MODEL_FIELDS})
    initialize(model, args.seed)
    manifest = dict(config=config, activation=activation, device='cpu',
                    parameters=sum(p.numel() for p in model.parameters()),
                    ffn_parameters=sum(p.numel() for p in model.blocks[0].ffn.parameters()),
                    effective_hidden_width=(2*config['d_ff'])//3 if activation=='swiglu' else config['d_ff'],
                    common_initial_sha256=fingerprint_parameters(model, True),
                    all_initial_sha256=fingerprint_parameters(model),
                    data_sha256=data_hashes, source_sha256=source_hashes,
                    train_sampling_seed=args.seed+1, validation_seed=args.seed+2,
                    train_evaluation_seed=args.seed+3, status='running')
    assert manifest['parameters']==14_556_288
    assert manifest['ffn_parameters']==1_179_648
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    train_rng = torch.Generator().manual_seed(args.seed+1)
    started = time.perf_counter()
    with (output/'loss.csv').open('x', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['step','train_loss','train_eval_loss','val_loss','elapsed_seconds'])
        for step in range(1, args.steps+1):
            model.train()
            x,y = batch(data[0], config, train_rng, 'cpu')
            optimizer.zero_grad(set_to_none=True)
            train_loss = loss(model,x,y)
            if not torch.isfinite(train_loss):
                raise ValueError(f'Nonfinite training loss: {activation}, {step}')
            train_loss.backward()
            optimizer.step()
            if step==1 or step%args.eval_interval==0 or step==args.steps:
                val = evaluate(model,data[1],config,args.seed+2)
                train_eval = evaluate(model,data[0],config,args.seed+3)
                if not all(torch.isfinite(torch.tensor(v)) for v in (val,train_eval)):
                    raise ValueError('Nonfinite evaluation')
                row=[step,train_loss.item(),train_eval,val,time.perf_counter()-started]
                writer.writerow(row); f.flush()
                print(activation,*row,flush=True)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                    step=step, torch_rng_state=torch.get_rng_state(), train_rng_state=train_rng.get_state()),
               output/'checkpoint.pt')
    write_samples(model,config,args.data_dir,output)
    manifest.update(status='completed', checkpoint_sha256=digest(output/'checkpoint.pt'))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--eval-interval',type=int,default=250)
    p.add_argument('--eval-batches',type=int,default=20)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if min(args.steps,args.eval_interval,args.eval_batches,args.threads)<1:
        p.error('numeric settings must be positive')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    suite=args.output or ROOT/'results/ffn_comparison'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_seed')/str(args.seed)
    suite.mkdir(parents=True,exist_ok=False)
    source_dir=suite/'source'; source_dir.mkdir()
    for filename in SOURCES:
        (source_dir/filename).write_bytes((ROOT/filename).read_bytes())
    source_hashes={f:digest(source_dir/f) for f in SOURCES}
    paths=[args.data_dir/f for f in ('train_tokens.pt','validation_tokens.pt')]
    data=[torch.load(f,map_location='cpu',weights_only=True) for f in paths]
    for ids in data:
        if ids.ndim!=1 or ids.dtype!=torch.long or len(ids)<=BASE['context_length'] or ids.min()<0 or ids.max()>=BASE['vocab_size']:
            raise ValueError('Invalid token tensor')
    hashes={f.name:digest(f) for f in [*paths,args.data_dir/'bpe_merges.pkl']}
    protocol=dict(status='running',seed=args.seed,steps=args.steps,eval_interval=args.eval_interval,
                  eval_batches=args.eval_batches,threads=args.threads,device='cpu',
                  python=platform.python_version(),torch=torch.__version__,platform=platform.platform(),
                  source_sha256=source_hashes,data_sha256=hashes,
                  git_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  initialization='Independent stable SHA-256-derived seed per named module; common modules match across all designs.',
                  comparison='All FFNs bias=False; 1536 ordinary width vs 1024 SwiGLU width. Same matrix/total parameter budget.',
                  limits='One seed; sampled fixed train/validation windows, not complete datasets; sequential timings are not controlled speed benchmarks.')
    (suite/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('SUITE',suite.resolve(),flush=True)
    try:
        manifests=[run(args,a,data,hashes,source_hashes,suite) for a in ACTIVATIONS]
        assert len({m['common_initial_sha256'] for m in manifests})==1
        assert len({m['all_initial_sha256'] for m in manifests[:3]})==1
        protocol['status']='training_completed'
        (suite/'protocol.json').write_text(json.dumps(protocol,indent=2))
        from summarize_ffn import summarize
        summarize(suite)
        protocol['status']='completed'
        (suite/'protocol.json').write_text(json.dumps(protocol,indent=2))
    except Exception as error:
        protocol.update(status='failed',error=repr(error))
        (suite/'protocol.json').write_text(json.dumps(protocol,indent=2))
        raise
if __name__=='__main__': main()
