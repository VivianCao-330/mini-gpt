
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT
from batches import get_batch


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


CONFIG = {
    "vocab_size": 513,
    "d_model": 384,
    "n_heads": 6,
    "d_ff": 1536,
    "n_layers": 8,
    "batch_size": 4,
    "context_length": 128,
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "max_steps": 3000,
    "eval_interval": 250,
    "eval_batches": 20,
}


def calculate_loss(model, x, y):
    logits = model(x)
    V = logits.size(-1)

    loss = F.cross_entropy(
        logits.reshape(-1, V),
        y.reshape(-1)
    )

    return loss


def evaluate(model, val_ids, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(CONFIG["eval_batches"]):
            x, y = get_batch(
                val_ids,
                CONFIG["batch_size"],
                CONFIG["context_length"],
                device
            )

            loss = calculate_loss(model, x, y)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    torch.manual_seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    train_ids = torch.load(
        DATA_DIR / "train_tokens.pt",
        map_location="cpu",
        weights_only=True
    )

    val_ids = torch.load(
        DATA_DIR / "validation_tokens.pt",
        map_location="cpu",
        weights_only=True
    )

    model = GPT(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        d_ff=CONFIG["d_ff"],
        n_layers=CONFIG["n_layers"]
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())

    print("Device:", device)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    log_path = RESULTS_DIR / "baseline.csv"
    checkpoint_path = CHECKPOINT_DIR / "baseline.pt"

    start_step = 0
    previous_elapsed = 0.0

    # Restore training state if a checkpoint exists.
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        saved_config = checkpoint["config"]

        for key in [
            "vocab_size",
            "d_model",
            "n_heads",
            "d_ff",
            "n_layers",
            "batch_size",
            "context_length",
            "learning_rate",
            "weight_decay"
        ]:
            if saved_config[key] != CONFIG[key]:
                raise ValueError(
                    f"Configuration mismatch: {key}"
                )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_step = checkpoint["step"]
        print(f"Resuming from step {start_step}")

        if not log_path.is_file():
            raise FileNotFoundError(
                "Checkpoint exists, but baseline.csv is missing"
            )

        with log_path.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))

        if not rows:
            raise ValueError("Existing CSV contains no training rows")

        if int(rows[-1]["step"]) != start_step:
            raise ValueError(
                "Checkpoint and CSV steps do not match"
            )

        previous_elapsed = float(
            rows[-1]["elapsed_seconds"]
        )

    else:
        print("No checkpoint found. Starting fresh training.")

    # This is outside the if/else, so it runs in both cases.
    start_time = time.perf_counter()

    mode = "a" if start_step > 0 else "w"

    with log_path.open(
        mode, newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if start_step == 0:
            writer.writerow([
                "step",
                "train_loss",
                "val_loss",
                "elapsed_seconds"
            ])

        for step in range(
            start_step + 1,
            CONFIG["max_steps"] + 1
        ):
            model.train()

            x, y = get_batch(
                train_ids,
                CONFIG["batch_size"],
                CONFIG["context_length"],
                device
            )

            optimizer.zero_grad()

            loss = calculate_loss(model, x, y)

            loss.backward()
            optimizer.step()

            if (
                step == 1
                or step % CONFIG["eval_interval"] == 0
                or step == CONFIG["max_steps"]
            ):
                train_loss = loss.item()
                val_loss = evaluate(model, val_ids, device)

                if device.type == "mps":
                    torch.mps.synchronize()

                elapsed = (
                    previous_elapsed
                    + time.perf_counter()
                    - start_time
                )

                print(
                    f"Step {step}: "
                    f"train={train_loss:.4f}, "
                    f"val={val_loss:.4f}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True
                )

                writer.writerow([
                    step,
                    train_loss,
                    val_loss,
                    elapsed
                ])
                file.flush()

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": CONFIG
                }

                torch.save(checkpoint, checkpoint_path)

    print("Training run finished!")
    print("Log saved:", log_path)
    print("Checkpoint saved:", checkpoint_path)


if __name__ == "__main__":
    main()
