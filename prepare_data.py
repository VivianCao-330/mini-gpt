from pathlib import Path
import json
from datasets import load_dataset
ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"
DATA_DIR.mkdir(exist_ok=True)
def save_stories(split, max_stories):
  dataset = load_dataset(
    "roneneldan/TinyStories",
    split=split,
    revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    streaming=True
)
  output_path = DATA_DIR / f"{split}.jsonl"
  if output_path.exists():
    raise FileExistsError(f"Refusing to overwrite {output_path}")
  saved = 0
  with output_path.open("w", encoding="utf-8") as file:
    for example in dataset:
      story = example["text"].strip()
      if not story:
        continue
      file.write(json.dumps({"text": story}, ensure_ascii=False)+ "\n")  
      saved += 1
      if saved >= max_stories:
        break
  print(f"{split}: saved {saved} stories")
  print(f"File: {output_path}")
  assert saved == max_stories
if __name__ == "__main__":
  save_stories("train", 10000)
  save_stories("validation", 1000)
