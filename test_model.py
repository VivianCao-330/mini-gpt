import torch
from model import CausalSelfAttention

attention = CausalSelfAttention(d_model=8, n_heads=2)
x = torch.randn(1, 4, 8, requires_grad=True)

y = attention(x)

assert y.shape == (1, 4, 8)
assert torch.isfinite(y).all()

y.sum().backward()

assert x.grad is not None
assert torch.isfinite(x.grad).all()

print("Basic multi-head attention test passed!")
attention.eval()

x1 = torch.randn(1, 4, 8)
x2 = x1.clone()

# Change the final token dramatically
x2[:, 3, :] += 100

with torch.no_grad():
    y1 = attention(x1)
    y2 = attention(x2)

# Earlier token outputs must be unaffected
assert torch.allclose(y1[:, :3, :], y2[:, :3, :], atol=1e-6)

print("Causality test passed!")
import torch
from model import RoPE

rope = RoPE(head_dim=8)

x = torch.randn(2, 4, 5, 8)
y = rope(x)

assert y.shape == x.shape
assert torch.isfinite(y).all()

# Rotation preserves Euclidean norm.
assert torch.allclose(
    torch.linalg.vector_norm(x, dim=-1),
    torch.linalg.vector_norm(y, dim=-1),
    atol=1e-5
)

# Position zero is unchanged.
assert torch.allclose(
    y[:, :, 0, :],
    x[:, :, 0, :],
    atol=1e-6
)

print("RoPE tests passed!")
from model import FeedForward
ffn = FeedForward(d_model=4, d_ff=8)

x = torch.randn(2, 3, 4)
output = ffn(x)

print("FFN input shape:", x.shape)
print("FFN output shape:", output.shape)

assert output.shape == (2, 3, 4)
assert torch.isfinite(output).all()

loss = output.square().mean()
loss.backward()

assert ffn.fc1.weight.grad is not None
assert ffn.fc2.weight.grad is not None

print("FFN tests passed!")
from model import TransformerBlock

block = TransformerBlock(d_model=8, n_heads=2, d_ff=32)

x = torch.randn(2, 4, 8, requires_grad=True)
y = block(x)

assert y.shape == x.shape
assert torch.isfinite(y).all()

y.square().mean().backward()

assert x.grad is not None
assert torch.isfinite(x.grad).all()

print("Transformer block tests passed!")
from model import TransformerBlock
import torch

block = TransformerBlock(d_model=8, n_heads=2, d_ff=32)

x = torch.randn(2, 4, 8, requires_grad=True)
y = block(x)

assert y.shape == x.shape
assert torch.isfinite(y).all()

y.square().mean().backward()

assert x.grad is not None
assert torch.isfinite(x.grad).all()

print("Transformer block test passed!")
import torch
from model import GPT

model = GPT(
    vocab_size=271,
    d_model=8,
    n_heads=2,
    d_ff=32,
    n_layers=2
)

for batch_size in [1, 2, 8]:
    token_ids = torch.randint(0, 271, (batch_size, 4))

    logits = model(token_ids)

    assert logits.shape == (batch_size, 4, 271)
    assert torch.isfinite(logits).all()

    print(f"Batch size {batch_size}: {logits.shape}")

token_ids = torch.randint(0, 271, (2, 4))
logits = model(token_ids)

loss = logits.square().mean()
loss.backward()

assert model.embedding.weight.grad is not None
assert model.lm_head.weight.grad is not None

print("GPT forward and backward tests passed!")
