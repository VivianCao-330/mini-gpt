import json
import pickle
from pathlib import Path
import torch
from tokenizer import train_bpe, encode
ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"
BPE_VOCAB_SIZE=512
BPE_TRAIN_STORIES=200

def read_stories(path):
    stories=[]
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            example=json.loads(line)
            stories.append(example["text"])
    return stories

def encode_stories(stories,merges,eos_id):
    all_ids=[]
    for story in stories:
        story_ids=encode(story,merges)
        all_ids.extend(story_ids)
        all_ids.append(eos_id)
    return torch.tensor(all_ids,dtype=torch.long)
def main():
    for name in ("train_tokens.pt", "validation_tokens.pt", "bpe_merges.pkl"):
        if (DATA_DIR/name).exists():
            raise FileExistsError(f"Refusing to overwrite {DATA_DIR/name}")
    train_stories=read_stories(DATA_DIR/"train.jsonl")
    val_stories=read_stories(DATA_DIR/"validation.jsonl")
    print("Training BPE...")
    bpe_text="\n".join(train_stories[:BPE_TRAIN_STORIES])
    merges=train_bpe(
        bpe_text,
        vocab_size=BPE_VOCAB_SIZE
    )
    eos_id=256+len(merges)
    actual_vocab_size=eos_id+1
    print("BPE merges:", len(merges))
    print("EOS token ID",eos_id)
    print("Model vocabulary size:", actual_vocab_size)

    train_ids=encode_stories(train_stories,merges,eos_id)
    val_ids=encode_stories(val_stories,merges,eos_id)

    assert train_ids.min().item() >= 0
    assert train_ids.max().item() < actual_vocab_size
    assert val_ids.min().item()>=0
    assert val_ids.max().item()<actual_vocab_size

    torch.save(train_ids, DATA_DIR / "train_tokens.pt")
    torch.save(val_ids, DATA_DIR / "validation_tokens.pt")

    with(DATA_DIR/"bpe_merges.pkl").open("wb") as file:
        pickle.dump(merges,file)
    print("Training tokens:", len(train_ids))
    print("Validation tokens:", len(val_ids))
    print("Token datasets saved!")


if __name__ == "__main__":
    main()