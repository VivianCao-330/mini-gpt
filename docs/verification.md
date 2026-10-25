# Verification record

Executed locally on 2026-10-26 (Asia/Shanghai), Python 3.13.15, PyTorch 2.14.1, CPU with four threads for formal training.

- All four formal models completed 3000 steps. Each manifest records `completed`, 14,571,648 parameters, matching source/data hashes, and its own checkpoint hash.
- Each formal checkpoint strictly loads, and recomputing the 20 fixed validation batches exactly matches its final CSV loss.
- Original baseline remains at step 3000. Original checkpoint and CSV fingerprints are recorded in `data/provenance.json`; original files were never rewritten.
- Model tests passed: shapes, finite outputs/gradients, causal masking, RoPE rotation norm/position-zero behavior, and next-token cross entropy.
- Experiment tests passed: matching parameter initialization for every variant, finite gradients, causal behavior, expected activation/RoPE settings, deterministic windows, shifted targets, and immediate EOS stopping after one forward call.
- Two independent two-step tiny suites produced exactly equal training/validation losses for all four variants. These are verification only and excluded from the formal result table.
- Tokenizer round-trip, Unicode, merge, special-token, and save/load checks passed.
- Preprocessing was rerun in a clean scratch copy from the existing raw JSONL files. It reconstructed both token tensors and BPE pickle with exact SHA-256 matches to the historical artifacts. All five raw/tokenizer/tensor fingerprints matched.
- All Python sources compile; training wrapper, experiment CLI, and sampling CLI help commands execute.
- Historical and all treatment checkpoints generated text successfully using seed 42, a 40-token budget, and CPU. Samples remain imperfect; no coherence claim is made.
- Matplotlib generated real historical and controlled CSV curves in PNG/SVG. The architecture image and controlled plot were visually inspected for labels/layout.

The pinned remote dataset download was not repeated; existing local raw files were used for preprocessing verification. The cached revision is documented with its provenance uncertainty. Cross-device bitwise equality, multi-seed significance, full-validation loss, and accelerator execution were not verified.
