"""Generate using either historical or controlled checkpoints."""
import argparse
import pickle
from pathlib import Path
import torch
from model import GPT
from generate import generate
from tokenizer import encode, build_vocab
from experiments import MODEL_KEYS
ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/baseline.pt')
    p.add_argument('--data-dir', type=Path, default=ROOT/'data')
    p.add_argument('--prompt', default='Once upon a time')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-new-tokens', type=int, default=100)
    p.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    a = p.parse_args()
    if not a.prompt or a.max_new_tokens < 0: p.error('nonempty prompt and nonnegative token count required')
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    # Load only trusted, locally built pickle files.
    with (a.data_dir/'bpe_merges.pkl').open('rb') as f: merges = pickle.load(f)
    vocab = build_vocab(merges)
    eos = 256 + len(merges)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    c = dict(use_rope=True, activation='relu', **{k:v for k,v in checkpoint['config'].items() if k not in ('use_rope','activation')})
    c.update({k:checkpoint['config'][k] for k in ('use_rope','activation') if k in checkpoint['config']})
    if eos + 1 != c['vocab_size']: raise ValueError('Tokenizer/checkpoint vocabulary mismatch')
    model = GPT(**{k:c[k] for k in MODEL_KEYS}).to(a.device)
    model.load_state_dict(checkpoint['model'])
    prompt = torch.tensor([encode(a.prompt, merges)], dtype=torch.long, device=a.device)
    ids = generate(model, prompt, a.max_new_tokens, c['context_length'], eos)[0].tolist()
    if ids[-1] == eos: ids.pop()
    # Sampling can end mid UTF-8 sequence; display the replacement character.
    print(b''.join(vocab[i] for i in ids).decode('utf-8', errors='replace'))
if __name__ == '__main__': main()
