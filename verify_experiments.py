"""Recompute final validation losses and verify all local checkpoints."""
import sys,json,csv,torch
from pathlib import Path
from experiments import MODEL_KEYS,loss,batch,digest
from model import GPT
torch.set_num_threads(4)
root=Path(__file__).resolve().parent
data=torch.load(root/'data/validation_tokens.pt',weights_only=True,map_location='cpu')
for folder in sorted((root/'results/controlled').iterdir()):
 c=torch.load(folder/'checkpoint.pt',weights_only=True,map_location='cpu'); conf=c['config']; assert c['step']==3000
 m=GPT(**{k:conf[k] for k in MODEL_KEYS});m.load_state_dict(c['model'],strict=True);m.eval()
 rng=torch.Generator().manual_seed(44)
 with torch.no_grad(): v=sum(loss(m,*batch(data,conf,rng,'cpu')).item() for _ in range(20))/20
 rows=list(csv.DictReader((folder/'loss.csv').open()));assert abs(v-float(rows[-1]['val_loss']))<1e-7,(folder,v)
 print(folder.name,'step 3000, strict load, validation recomputation MATCH',v,flush=True)
 del m,c
c=torch.load(root/'checkpoints/baseline.pt',weights_only=True,map_location='cpu');assert c['step']==3000
m=GPT(**{k:c['config'][k] for k in MODEL_KEYS if k not in ('use_rope','activation')});m.load_state_dict(c['model'],strict=True)
p=json.loads((root/'data/provenance.json').read_text())['files']
assert digest(root/'checkpoints/baseline.pt')==p['checkpoints/baseline.pt']['sha256']
assert digest(root/'results/baseline.csv')==p['results/baseline.csv']['sha256']
print('Historical baseline step, strict load, checkpoint and CSV SHA-256: unchanged',flush=True)
