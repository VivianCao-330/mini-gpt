#overfitting test
import torch
import torch.nn.functional as F
from model import GPT
torch.manual_seed(42)
model=GPT(
    vocab_size=271,
    d_model=32,
    n_heads=4,
    d_ff=128,
    n_layers=2
)
optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0
)
sequence = torch.tensor([
    [10, 20, 30, 40, 50, 60],
    [15, 25, 35, 45, 55, 65]
])
inputs=sequence[:,:-1]
targets=sequence[:,1:]
model.train()
for step in range(101):
    optimizer.zero_grad()
    logits=model(inputs)
    B,T,V=logits.shape
    loss=F.cross_entropy(
        logits.reshape(B*T,V),
        targets.reshape(B*T)
    )
    loss.backward()
    optimizer.step()
    if step % 20 ==0:
        print(f"Step{step}:Loss:{loss.item():4f} ")
#evaluation
model.eval()
val_sequence=torch.tensor([
    [12, 22, 32, 42, 52, 62],
    [17, 27, 37, 47, 57, 67]
])
val_inputs=val_sequence[:,:-1]
val_targets=val_sequence[:,1:]
with torch.no_grad():
    val_logits=model(val_inputs)
    B,T,V=val_logits.shape
    val_loss=F.cross_entropy(
        val_logits.reshape(B*T,V),
        val_targets.reshape(B*T)
    )
    print(f"Validation Loss:{val_loss.item():4f}")
#verifying MPS availability
import torch
print("MPS available:", torch.backends.mps.is_available())
#saving parameters
torch.save(model.state_dict(),"gpt_smoke_test.pt")
loaded_model=GPT(
    vocab_size=271,
    d_model=32,
    n_heads=4,
    d_ff=128,
    n_layers=2
)
weights=torch.load(
    "gpt_smoke_test.pt",
    map_location="cpu",
    weights_only=True
)
loaded_model.load_state_dict(weights)
loaded_model.eval()
with torch.no_grad():
    original_logits=model(val_inputs)
    loaded_logits=loaded_model(val_inputs)
assert torch.allclose(original_logits,loaded_logits,atol=1e-6)
print("Checkpoint save/load test passed!")
