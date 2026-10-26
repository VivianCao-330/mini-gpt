"""Verify saved FFN checkpoints, summarize real CSVs, and produce a report."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import torch
from model import GPT
from experiments import batch, digest, loss

ACTIVATIONS=('relu','gelu','silu','swiglu')
FIELDS=('vocab_size','d_model','n_heads','d_ff','n_layers','use_rope','activation','ffn_bias')

def summarize(suite):
    suite=Path(suite)
    protocol=json.loads((suite/'protocol.json').read_text())
    torch.set_num_threads(protocol['threads'])
    for name,expected in protocol['source_sha256'].items():
        assert digest(suite/'source'/name)==expected, f'Source snapshot changed: {name}'
    for name in ('model.py','experiments.py'):
        assert digest(Path(__file__).resolve().parent/name)==protocol['source_sha256'][name], 'Use the archived source to verify this suite after changing model code.'
    root=Path(__file__).resolve().parent
    ids=torch.load(root/'data/validation_tokens.pt',map_location='cpu',weights_only=True)
    assert digest(root/'data/validation_tokens.pt')==protocol['data_sha256']['validation_tokens.pt']
    summaries=[]; logs={}; manifests=[]
    for a in ACTIVATIONS:
        folder=suite/f'{a}_seed{protocol["seed"]}'
        m=json.loads((folder/'manifest.json').read_text()); manifests.append(m)
        assert m['status']=='completed'
        assert m['data_sha256']==protocol['data_sha256']
        assert m['source_sha256']==protocol['source_sha256']
        assert digest(folder/'checkpoint.pt')==m['checkpoint_sha256']
        with (folder/'loss.csv').open() as f:
            rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        expected_steps=sorted({1,protocol['steps'],*range(protocol['eval_interval'],protocol['steps']+1,protocol['eval_interval'])})
        assert [int(r['step']) for r in rows]==expected_steps
        assert all(math.isfinite(v) for r in rows for v in r.values())
        checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
        assert checkpoint['step']==protocol['steps'] and checkpoint['config']==m['config']
        config=checkpoint['config']
        model=GPT(**{k:config[k] for k in FIELDS}); model.load_state_dict(checkpoint['model'],strict=True); model.eval()
        assert sum(p.numel() for p in model.parameters())==m['parameters']==14_556_288
        rng=torch.Generator().manual_seed(m['validation_seed'])
        with torch.no_grad():
            checked=sum(loss(model,*batch(ids,config,rng,'cpu')).item() for _ in range(config['eval_batches']))/config['eval_batches']
        assert abs(checked-rows[-1]['val_loss'])<1e-7,(a,checked,rows[-1])
        last=rows[-1];best=min(rows,key=lambda r:r['val_loss']);logs[a]=rows
        summaries.append(dict(activation=a,steps=int(last['step']),parameters=m['parameters'],
                              hidden_width=m['effective_hidden_width'],ffn_bias=False,
                              final_train_minibatch=last['train_loss'],final_train_eval=last['train_eval_loss'],
                              final_val=last['val_loss'],best_recorded_val=best['val_loss'],best_step=int(best['step']),
                              val_perplexity=math.exp(last['val_loss']),elapsed_seconds=last['elapsed_seconds'],
                              final_checkpoint_validation_verified=True))
        del model,checkpoint
    assert len({m['common_initial_sha256'] for m in manifests})==1
    assert len({m['all_initial_sha256'] for m in manifests[:3]})==1
    for r in summaries:r['delta_val_vs_relu']=r['final_val']-summaries[0]['final_val']
    with (suite/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]),lineterminator='\n');w.writeheader();w.writerows(summaries)
    (suite/'verification.json').write_text(json.dumps(dict(checkpoints='all four hash and strict-load checks passed',
        validation='all four final CSV losses reproduced from checkpoints',
        common_initialization='matching outside FFN; all initial weights match for ReLU/GELU/SiLU',
        source_snapshots='hashes verified',data='validation tensor hash verified',runs=summaries),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=suite/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for a,rows in logs.items():
        for ax,key in zip(axes,('train_eval_loss','val_loss')):
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',ms=3,label=a)
    for ax,title in zip(axes,('Fixed training windows','Fixed validation windows')):
        ax.set(xlabel='Optimizer step',ylabel='Cross entropy (nats/token)',title=title);ax.grid(alpha=.2);ax.legend()
    fig.suptitle(f'FFN design comparison · seed {protocol["seed"]} · equal parameter budget · FFN biases off')
    fig.savefig(figures/'loss.png',dpi=180);fig.savefig(figures/'loss.svg');plt.close(fig)
    rows_html=''.join(f'<tr><td>{r["activation"]}</td><td>{r["final_train_eval"]:.5f}</td><td>{r["final_val"]:.5f}</td><td>{r["delta_val_vs_relu"]:+.5f}</td><td>{r["val_perplexity"]:.3f}</td><td>{r["parameters"]:,}</td></tr>' for r in summaries)
    samples=''
    for a in ACTIVATIONS:
        records=json.loads((suite/f'{a}_seed{protocol["seed"]}'/'samples.json').read_text())
        samples+=f'<h3>{a}</h3>'+''.join(f'<p><b>{html.escape(r["prompt"])}</b></p><pre>{html.escape(r["text"])}</pre>' for r in records)
    report=f'''<!doctype html><meta charset="utf-8"><title>mini-gpt FFN comparison</title>
<style>body{{max-width:1050px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#17212b}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:#f3f5f7;padding:16px}}</style>
<h1>mini-gpt: ReLU / GELU / SiLU / SwiGLU</h1>
<p>Actual fresh CPU runs: {protocol['steps']} updates, seed {protocol['seed']}. All FFN biases off. Ordinary hidden width 1536; SwiGLU hidden width 1024. Every model has 14,556,288 trainable parameters. Final checkpoint validation losses were independently recomputed and match the CSV records.</p>
<table><tr><th>FFN</th><th>Train eval</th><th>Validation</th><th>Δ vs ReLU</th><th>Token PPL</th><th>Parameters</th></tr>{rows_html}</table>
<img src="figures/loss.png" alt="Training and validation loss from actual CSVs">
<h2>Interpretation and limits</h2><p>Lower cross entropy is better. Deltas describe this one paired seed and do not establish statistical significance. ReLU/GELU/SiLU differ only in the activation. SwiGLU changes the FFN structure and hidden width while preserving its parameter budget; it is not an activation-only ablation. The shared embedding, attention, norms, and output head start with identical weights across all four models. Full FFN initialization matches for the three ordinary activations; differently shaped SwiGLU matrices cannot have identical tensors.</p>
<p>Every run uses the same training-window sequence and fixed evaluation windows. Train-eval and validation each average {protocol['eval_batches']} batches of 4 × 128 positions after an update. They are sampled windows, not full-corpus losses. Training-minibatch losses are recorded separately. AdamW learning rate 0.0003, weight decay 0.01, RoPE on. No scheduler, dropout, clipping, or accumulation. Timing is sequential elapsed time including evaluation and is not a controlled throughput benchmark.</p>
<p>Existing historical runs used a different bias/init protocol and are preserved separately; use this suite's fresh ReLU control for comparisons. Repeated seeds and full-validation evaluation are future work; no such results are claimed.</p>
<h2>Evidence</h2><p>See protocol.json; source/ for the exact runnable source snapshot; each run's manifest.json, loss.csv, checkpoint.pt, samples.json; summary.csv and verification.json. Checkpoints are local ignored artifacts. The manifest records status completed only after checkpoint and samples are written.</p>
<h2>Fixed-prompt samples</h2><p>Same three prompts and sampling seed 2026; maximum 100 new tokens, immediate EOS stop, temperature 1. These examples are illustrative, not a quality benchmark.</p>{samples}'''
    (suite/'report.html').write_text(report)
    print(json.dumps(summaries,indent=2),flush=True)
    return summaries

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',type=Path)
    summarize(p.parse_args().suite)
