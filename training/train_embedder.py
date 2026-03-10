import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = Path(__file__).parent / "pairs.jsonl"
OUT_DIR = Path(__file__).parent / "finetuned_model"

def load_pairs(path: Path):
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        q = obj["query"].strip()
        n = obj["note"].strip()
        if q and n:
            examples.append(InputExample(texts=[q, n]))
    return examples

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATA_PATH}")

    train_examples = load_pairs(DATA_PATH)
    if len(train_examples) < 10:
        raise ValueError("Add at least 10 pairs. Better: 30–100 pairs.")

    print(f"Loaded {len(train_examples)} training pairs")

    model = SentenceTransformer(BASE_MODEL)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=50,
        show_progress_bar=True,
        output_path=str(OUT_DIR),
    )

    print(f"✅ Saved finetuned model to: {OUT_DIR}")

if __name__ == "__main__":
    main()