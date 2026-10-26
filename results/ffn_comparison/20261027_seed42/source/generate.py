import torch
from model import GPT
def generate(
    model,
    prompt_ids,
    max_new_tokens,
    context_window,
    eos_id=None
):
    model.eval()

    if prompt_ids.shape[0] != 1:
        raise ValueError(
            "This EOS-aware generator currently supports batch size 1."
        )

    token_ids = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):

            context = token_ids[:, -context_window:]

            logits = model(context)

            next_logits = logits[:, -1, :]

            probabilities = torch.softmax(
                next_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            token_ids = torch.cat(
                [token_ids, next_token],
                dim=1
            )

            if eos_id is not None:
                if next_token.item() == eos_id:
                    break

    return token_ids

if __name__ == "__main__":
    torch.manual_seed(42)

    model = GPT(
        vocab_size=271,
        d_model=32,
        n_heads=4,
        d_ff=128,
        n_layers=2
    )

    weights = torch.load(
        "gpt_smoke_test.pt",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(weights)

    prompt = torch.tensor([[10, 20, 30]])

    generated = generate(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=5,
        context_window=16
    )

    print("Original prompt:", prompt.tolist())
    print("Generated sequence:", generated.tolist())

    assert generated.shape == (1, 8)
    assert torch.equal(generated[:, :3], prompt)

    print("Generation test passed!")