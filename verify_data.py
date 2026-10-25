"""Verify reproduced raw data and tensors against the recorded local dataset."""
import argparse,json,sys
from pathlib import Path
import torch
from experiments import digest, ROOT
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-dir',type=Path,default=ROOT/'data')
a=p.parse_args(); expected=json.loads((ROOT/'data/provenance.json').read_text())['files']; ok=True
for name,record in expected.items():
    if not name.startswith('data/'): continue
    file=a.data_dir/Path(name).name
    actual=digest(file)
    # Torch serialization bytes may vary by version. Hash mismatch is reported explicitly.
    same=actual==record['sha256']; ok &= same
    print(('MATCH' if same else 'MISMATCH'),file.name,actual)
sys.exit(0 if ok else 1)
