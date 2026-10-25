import torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"
def get_batch(data, batch_size, context_length, device):
    T=context_length
    max_start=len(data)-T-1
    if max_start<0:
        raise ValueError("Not enough tokens to create a batch")
    starts=torch.randint(
        low=0,
        high=max_start+1,
        size=(batch_size,)
    ).tolist()
    x=torch.stack([
        data[i:i+T]
        for i in starts
    ])
    y=torch.stack([
        data[i+1:i+T+1]
        for i in starts
    ])
    return x.to(device),y.to(device)
if __name__ == "__main__":
    train_ids=torch.load(
        DATA_DIR/"train_tokens.pt",
        map_location="cpu",
        weights_only=True
    )
    val_ids=torch.load(
        DATA_DIR/"validation_tokens.pt",
        map_location="cpu",
        weights_only=True
    )
    device=torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    x,y=get_batch(
        data=train_ids,
        batch_size=4,
        context_length=128,
        device=device
    )
    print("Device:", device)
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    print("Example inputs:", x[0, :10].tolist())
    print("Example targets:", y[0, :10].tolist())

    assert x.shape == (4, 128)
    assert y.shape == (4, 128)
    assert torch.equal(x[:, 1:], y[:, :-1])

    print("Batch test passed!")