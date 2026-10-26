# mini-gpt

A small, from-scratch decoder-only Transformer trained on TinyStories, with a byte-level BPE tokenizer and reproducible controlled ablations. Architecture: 8 layers, width 384, 6 heads, FFN width 1536, RMSNorm, RoPE, ReLU, and an untied language-model head; **14,571,648 parameters**.

The original 3000-step baseline and its CSV are preserved. Fresh controlled runs separately test RoPE-off, GELU, and zero weight decay against a fresh baseline. Read [actual results](docs/results.md), [technical note](docs/technical-note.md), and [architecture](docs/architecture.png).

## Setup

```sh
git clone https://github.com/VivianCao-330/mini-gpt.git
cd mini-gpt
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Tested with Python 3.13.15 and the exact package versions in requirements.txt on macOS ARM64, CPU, four threads. PyTorch accelerator availability depends on the host. The trainer chooses CUDA, then MPS, then CPU with `--device auto`; explicitly select `--device cpu` to match these runs. Strict deterministic mode may reject unsupported accelerator operations; use CPU in that case. Numerical results need not match across PyTorch versions or devices.

## Data and tokenizer

```sh
python prepare_data.py
python build_tokens.py
python verify_data.py
```

The downloader pins TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, streams in dataset order, and saves the first 10,000 nonempty training stories and 1,000 validation stories as JSONL. The tokenizer learns up to 256 BPE merges from the first 200 training stories, using regex pretokens over UTF-8 bytes. The 512 base/merged tokens plus EOS ID 512 make vocabulary size 513. EOS is appended after each story; tensors concatenate stories, so windows may cross boundaries. No validation text trains BPE. The recorded tensors contain 3,736,309 training and 335,543 validation tokens.

Raw data, tokens, pickle tokenizer, and checkpoints are intentionally excluded from git. Rebuild them locally. `data/provenance.json` records the original file hashes. `verify_data.py` exits nonzero on mismatches; investigate them before claiming matched reproduction. The historical downloader did not record revision, so the cached revision is evidence rather than proof of historical provenance. Serialization hashes can change across versions. Downloader/tokenizer refuse to overwrite existing artifacts; use a separate clean clone to rebuild. Only load pickle/tokenizer and checkpoint files you trust.

## Train a matched experiment suite

```sh
python experiments.py --variant all --seed 42 --steps 3000 \
  --eval-interval 250 --eval-batches 20 --device cpu --threads 4 \
  --output results/reproduction_seed42
```

`train_real.py` forwards to the same safe CLI. Each variant gets its own directory, CSV, manifest, and final checkpoint. Existing directories cause an error rather than overwrite or resume. Use a new output directory for every rerun. The committed `results/controlled/` directories already contain published logs, so the command above chooses a different root. To run only one treatment:

```sh
python experiments.py --variant rope_off --output results/new_rope_off
python experiments.py --variant gelu --output results/new_gelu
python experiments.py --variant no_weight_decay --output results/new_no_decay
```

Run matching controls with identical seed/data/steps when changing these settings. Shared defaults: AdamW lr 0.0003, decay 0.01, batch 4, context 128, 3000 updates, evaluation every 250 updates plus step 1, 20 fixed validation batches. Treatment changes only its named setting. Seed 42 controls initialization; independent generators 43 and 44 control training and validation sampling. Validation resets to the same windows at each evaluation; evaluation never changes training order. No dropout or stochastic model layers are used.

Checkpoints are written after the final update and contain model, optimizer, configuration, step, global CPU RNG, and training-sampler RNG. Automatic resume is deliberately unsupported in the controlled runner. Interrupted runs are incomplete and must be rerun under a new root. The manifest has status `completed` only after checkpoint writing. `train_legacy.py` is historical source retained for audit; it uses old resume behavior and should not be used for controlled comparisons.

## Generation

After training/rebuilding the tokenizer:

```sh
python sample_text.py \
  --checkpoint results/reproduction_seed42/baseline_control_seed42/checkpoint.pt \
  --prompt 'Once upon a time' --seed 42 --max-new-tokens 100 --device cpu
```

To sample the preserved historical baseline locally, select `--checkpoint checkpoints/baseline.pt`. Checkpoints are not supplied in a fresh Git clone. Generation samples from softmax with temperature 1, no top-k/top-p filtering, crops context to 128 tokens, and stops immediately when EOS is sampled. For generated incomplete UTF-8 sequences, display uses replacement characters. Changing the seed changes sampled text. Low loss alone does not establish coherent stories.

## Verification and plots

```sh
python -m unittest test_experiments
python test_model.py
python test_tokenizer.py
python test_loss.py
python experiments.py --tiny --steps 2 --eval-batches 1 --output results/verification_new
python verify_experiments.py --results-root results/reproduction_seed42
python plot_results.py --logs-only
python draw_architecture.py
```

`verify_experiments.py` recomputes final validation losses from local training checkpoints and compares them with CSVs. The preserved historical checkpoint is checked when available locally. `plot_results.py --logs-only` renders the committed published logs without requiring excluded checkpoints.

The tests verify causal attention, RoPE norm preservation, finite gradients, matching initial weights across treatments, repeatable batch selection, and immediate EOS termination. Tiny runs are smoke tests, never experimental evidence. `test_tokenizer.py` creates a disposable ignored local pickle. Historical `train_smoke.py` and the standalone `generate.py` demo target an old smoke checkpoint and are not required by this workflow.

`plot_results.py` reads the preserved CSV plus the formal `results/controlled/*/loss.csv`, checks completed manifests, checkpoint hashes, step counts, parameter counts and matching data/source fingerprints, and writes PNG/SVG plots, `results/summary.csv`, and `docs/results.md`. To rerender without local checkpoints, use `--logs-only`; that explicitly skips checkpoint verification. Default verification requires the ignored local final checkpoints. No fabricated or interpolated loss records are added.

## Files

- `model.py`: custom attention, adjacent-pair RoPE, RMSNorm, FFN, decoder.
- `tokenizer.py`, `prepare_data.py`, `build_tokens.py`, `batches.py`: data path.
- `experiments.py`, `train_real.py`: safe fresh-run controls.
- `sample_text.py`, `generate.py`: checkpoint sampling and EOS-aware loop.
- `results/baseline.csv`: immutable historical logs; `results/controlled/`: formal paired runs.
- `data/provenance.json`: original baseline/data hashes; per-run manifests: source/data/config metadata.
- `docs/`: plots, architecture source/image, technical note, experiment report, verification record.

No token data, checkpoint weights, credentials, environments, or model artifacts are committed. Logs and metadata are small and committed for inspection. Tiny/benchmark outputs are ignored. Sources are educational research code; dataset terms remain those of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
