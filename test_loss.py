import torch
import torch.nn.functional as F
from model import GPT
model=GPT(
    vocab_size=271,
    d_model=8,
    n_heads=2,
    d_ff=32,
    n_layers=2
)
sequence=torch.tensor([
    [10,20,30,40,50],
    [15,25,35,45,55]
])
inputs=sequence[:,:-1]
targets=sequence[:,1:]
logits=model(inputs)
B,T,V=logits.shape
loss=F.cross_entropy(logits.reshape(B*T,V),targets.reshape(B*T))
print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)
print("Logits shape:", logits.shape)
print("Cross-entropy loss:", loss.item())

assert torch.isfinite(loss)

loss.backward()

assert model.embedding.weight.grad is not None
assert model.lm_head.weight.grad is not None

print("Next-token cross-entropy test passed!")