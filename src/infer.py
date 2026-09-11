import argparse

import torch
from transformers import AutoTokenizer

from models import HierHateModel

FINE_LABELS = ["gender", "LGBT", "age", "region", "race", "religion", "socioeconomic", "etc"]
COARSE_LABELS = ["clean", "offensive", "hate"]


def predict(text, model, tokenizer, device, threshold=0.5, max_len=128):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    coarse_prob = torch.softmax(out["logits_coarse"], dim=-1)[0]
    fine_prob = torch.sigmoid(out["logits_fine"])[0]

    coarse_idx = int(torch.argmax(coarse_prob).item())
    fine_pred = [
        label for label, prob in zip(FINE_LABELS, fine_prob.tolist()) if prob >= threshold
    ]

    return {
        "coarse": COARSE_LABELS[coarse_idx],
        "coarse_confidence": float(coarse_prob[coarse_idx].item()),
        "fine_labels": fine_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plm", default="klue/bert-base")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.plm)
    model = HierHateModel(args.plm, num_coarse=3, num_fine=8)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    print(predict(args.text, model, tokenizer, device))


if __name__ == "__main__":
    main()
