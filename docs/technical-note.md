# mini-gpt technical note

## Implemented architecture

This is a decoder-only, pre-normalized Transformer with 14,571,648 trainable parameters. Token IDs [B,T] index a learned 513×384 embedding. Eight blocks apply `x = x + attention(RMSNorm(x))`, then `x = x + FFN(RMSNorm(x))`. A final RMSNorm and an independent, untied 384×513 bias-free linear head produce logits [B,T,513]. There are no learned absolute position embeddings, no dropout, and no embedding/head weight tying.

Each attention layer uses six heads of dimension 64. Q/K/V/output projections have no biases. Adjacent even/odd Q and K coordinates are rotated by RoPE (base 10000); V is unchanged. Attention scales QKᵀ by √64 and masks future positions before softmax. The FFN uses biased linear layers 384→1536→384 and ReLU by default. RMSNorm uses learned scale and epsilon 1e-8. Cross entropy predicts the next token at every context position, without padding or ignored targets.

## Training and experimental control

AdamW uses learning rate 3e-4, default betas (0.9,0.999), epsilon 1e-8, and decay 0.01 applied to all parameters, including norms and biases. No schedule, warmup, gradient clipping, or accumulation is used. Each update samples four windows of 128 tokens with replacement from the concatenated training tensor. 3000 updates process 1,536,000 target-token positions, including repeats; this is not one complete pass through the 3,736,309-token training corpus.

The paired suite starts all models from seed 42 and exactly matching parameter tensors. An independent CPU generator seeded 43 draws training windows, isolating training order from evaluation. At every evaluation, generator 44 resets to the same 20 validation batches (10,240 positions). Thus validation windows match across steps and variants, but can overlap and are not the full validation set. Each variant changes exactly one factor: remove Q/K RoPE without adding another position mechanism; replace ReLU with exact GELU; or set AdamW weight decay to zero. All other hyperparameters and data tensors match.

The historical baseline is retained separately. It resumed at steps 10 and 200, resetting the global RNG without restoring its state, and its early evaluation batch counts differed. Its recorded validation samples also changed between evaluations. It is therefore descriptive provenance rather than the causal comparator for the new ablations. Historical elapsed time is taken from its CSV, not claimed as a comparable speed benchmark.

## Reproduction limits

Each formal run records configuration, Python/PyTorch versions, device, source/data SHA-256 hashes, parameter count, and checkpoint hash. CPU deterministic algorithms are enabled and CPU thread count is fixed to four. Cross-version and cross-device bitwise equality is not promised. Final checkpoint includes optimizer, global CPU RNG, and sampling-generator state. The controlled trainer deliberately starts fresh and refuses existing output directories; automatic resume is not implemented. A failed run retains `status: running` and is never reported as complete.

Historical data download did not record a dataset revision. The revised downloader pins the revision found in the local Hugging Face cache; raw-file and tensor fingerprints in `data/provenance.json` are the authoritative match check. Hash mismatch must be investigated before treating regenerated data as equivalent. PyTorch serialization container bytes can differ between versions even for equal tensors.

## Interpretation limits and next work

Train loss is the last sampled training minibatch before its update. Validation loss is measured after that update, averaged over fixed held-out windows. Their difference is not a robust generalization-gap estimate. Token perplexity is exp(validation loss), specific to this tokenizer and sample. One seed cannot establish statistical significance or broad superiority; fixed windows may favor particular model behavior. Repeated seeds and full-validation evaluation remain planned.

The three research papers remain scheduled for Week 2 Day 5 at the user's direction. Foundation Gate is deferred until teachers are available. Neither task is marked complete.
