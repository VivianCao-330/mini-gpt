"""Render real CSV logs; summarize only verified completed formal runs."""
import csv,json,math,argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import ROOT, VARIANTS, digest
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--logs-only',action='store_true',help='Skip local checkpoint existence/hash verification')
args=p.parse_args()
D=ROOT/'docs'; D.mkdir(exist_ok=True)
def read(p):
    with p.open() as f: rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    assert rows and all(math.isfinite(v) for row in rows for v in row.values())
    assert all(a['step']<b['step'] for a,b in zip(rows,rows[1:]))
    return rows
old=read(ROOT/'results/baseline.csv')
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
for key,label in [('train_loss','Last training minibatch'),('val_loss','Sampled validation mean')]:
    ax.plot([r['step'] for r in old],[r[key] for r in old],marker='o',ms=3,label=label)
ax.set(xlabel='Optimizer step',ylabel='Cross entropy (nats / token)',title='Historical 3000-step baseline · changing validation samples')
ax.legend(); ax.grid(alpha=.2)
fig.savefig(D/'historical-loss.png',dpi=180); fig.savefig(D/'historical-loss.svg'); plt.close(fig)
records=[]
for name in VARIANTS:
    folder=ROOT/'results/controlled'/f'{name}_seed42'
    m=json.loads((folder/'manifest.json').read_text())
    assert m['status']=='completed',f'{name} is not complete'
    if not args.logs_only: assert digest(folder/'checkpoint.pt')==m['checkpoint_sha256']
    rows=read(folder/'loss.csv'); assert rows[-1]['step']==3000 and len(rows)==13
    records.append((name,m,rows))
reference=records[0][1]
for name,m,_ in records:
    assert m['data_sha256']==reference['data_sha256']
    assert m['source_sha256']==reference['source_sha256']
    assert m['parameters']==14571648
fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for name,m,rows in records:
    for ax,key in zip(axs,['train_loss','val_loss']):
        ax.plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',ms=3,label=name)
for ax,title in zip(axs,['Training · last minibatch','Validation · same 20 batches at every point']):
    ax.set(xlabel='Optimizer step',ylabel='Cross entropy (nats / token)',title=title); ax.grid(alpha=.2)
axs[1].legend(fontsize=8)
fig.suptitle('mini-gpt · controlled fresh starts · seed 42')
fig.savefig(D/'controlled-loss.png',dpi=180); fig.savefig(D/'controlled-loss.svg'); plt.close(fig)
base=records[0][2][-1]['val_loss']
summary=[]
for name,m,rows in records:
    last=rows[-1]; best=min(rows,key=lambda r:r['val_loss'])
    summary.append(dict(variant=name,status='completed',step=int(last['step']),train_loss=last['train_loss'],val_loss=last['val_loss'],best_val_loss=best['val_loss'],best_step=int(best['step']),delta_vs_control=last['val_loss']-base,perplexity=math.exp(last['val_loss']),seconds=last['elapsed_seconds']))
with (ROOT/'results/summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
lines=['# Experimental results','', f"All four formal runs completed 3000 fresh-start updates on CPU. Values are read directly from tracked CSVs; {'checkpoint verification skipped (--logs-only).' if args.logs_only else 'final checkpoints and hashes verified.'}",'', '| Variant | Final train | Final validation | Δ validation vs control | Best validation (step) | Token PPL |','|---|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['variant']} | {r['train_loss']:.5f} | {r['val_loss']:.5f} | {r['delta_vs_control']:+.5f} | {r['best_val_loss']:.5f} ({r['best_step']}) | {r['perplexity']:.3f} |")
lines += ['', '![Controlled training and validation loss](controlled-loss.png)', '',
'Historical baseline: final train 2.14841; final validation 2.12550; lowest recorded validation 2.11852 at step 2750. The CSV retains all 23 historical rows unchanged. Its resumed RNG and validation protocol differ from the controlled suite, so use the fresh control for ablation comparisons.', '',
'![Historical baseline loss](historical-loss.png)', '',
'## Analysis', '',
f"RoPE-off finishes {summary[1]['delta_vs_control']:+.5f} nats/token above the matched control, a substantial deterioration in this run. GELU changes final validation loss by {summary[2]['delta_vs_control']:+.5f}; zero decay changes it by {summary[3]['delta_vs_control']:+.5f}. These small latter differences are insufficient to claim a reliable improvement or harm from one seed. Retaining RoPE is supported by this paired run; activation and weight-decay choices remain unresolved without replication.", '',
'Positive validation deltas mean worse loss than the matched control. Disabling RoPE removes explicit positional encoding from Q/K; causal masking remains. GELU changes only the FFN nonlinearity. Zero weight decay changes only the AdamW decay coefficient. These comparisons hold data, initial weights, sampled training windows, evaluation windows, steps, and optimizer settings constant.', '',
'One paired seed and 20 fixed validation minibatches do not support statistical significance claims. Training curves report individual minibatches and are noisy. Validation uses overlapping windows sampled with replacement, not the entire held-out corpus. Do not infer a robust train/validation gap from these mismatched estimators. Perplexity is token-level exp(loss) for this 513-token vocabulary.', '',
'## Experiment status', '',
'- Completed: preserved historical baseline; fresh control; RoPE-off, GELU, and no-weight-decay runs; plots and checkpoint verification.',
'- Verification only: tiny two-step runs, repeatability runs, and ten-step full-model benchmark, excluded from this table.',
'- Planned: repeated seeds and full-validation evaluation. No results claimed.',
'Run `python plot_results.py` to regenerate this report, summary.csv, and both plot formats from the raw logs and local checkpoints.']
(D/'results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
