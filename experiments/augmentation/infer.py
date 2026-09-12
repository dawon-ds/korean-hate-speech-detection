import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.config import load_config
from src.models import FlatHateModel, HierHateModel

COARSE_ID2LABEL = ["clean", "offensive", "hate"]
FINE_ID2LABEL = [
    "gender",
    "LGBT",
    "age",
    "region",
    "race",
    "religion",
    "socioeconomic",
    "etc",
]

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config" / "base.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive inference for the flat or hierarchical hate-speech model."
    )
    parser.add_argument(
        "--model-type",
        choices=["flat", "hier"],
        default="flat",
        help="Model architecture to load.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to checkpoints/flat_best.pt or hier_best.pt inside the augmentation experiment.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fine-label sigmoid threshold.",
    )
    return parser.parse_args()


def build_model_and_tokenizer(cfg, device, model_type):
    plm_name = cfg.get("model", "plm_name")
    num_coarse = cfg.get("model", "num_coarse", default=3)
    num_fine = cfg.get("model", "num_fine", default=8)
    lambda_fine = cfg.get("model", "lambda_fine", default=1.0)
    lambda_hier = cfg.get("model", "lambda_hier", default=1.0)

    if model_type == "hier":
        model = HierHateModel(
            plm_name=plm_name,
            num_coarse=num_coarse,
            num_fine=num_fine,
            lambda_fine=lambda_fine,
            lambda_hier=lambda_hier,
        )
    else:
        model = FlatHateModel(
            plm_name=plm_name,
            num_coarse=num_coarse,
            num_fine=num_fine,
            lambda_fine=lambda_fine,
        )

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(plm_name)
    return model, tokenizer


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_text(text, model, tokenizer, device, threshold=0.5):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits_coarse = outputs["logits_coarse"]
    logits_fine = outputs["logits_fine"]

    prob_coarse = torch.softmax(logits_coarse, dim=-1)[0].cpu().tolist()
    pred_coarse_id = int(torch.argmax(logits_coarse, dim=-1)[0].cpu().item())
    pred_coarse_label = COARSE_ID2LABEL[pred_coarse_id]

    prob_fine = torch.sigmoid(logits_fine)[0].cpu().tolist()
    pred_fine_ids = [i for i, p in enumerate(prob_fine) if p >= threshold]
    pred_fine_labels = [FINE_ID2LABEL[i] for i in pred_fine_ids]

    return {
        "coarse_pred": pred_coarse_label,
        "coarse_prob": prob_coarse,
        "fine_pred": pred_fine_labels,
        "fine_prob": prob_fine,
    }


def main():
    args = parse_args()
    cfg = load_config(str(args.config.resolve()))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint_name = "hier_best.pt" if args.model_type == "hier" else "flat_best.pt"
        checkpoint = EXPERIMENT_DIR / "checkpoints" / checkpoint_name
    elif not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train the selected model first or pass --checkpoint."
        )

    model, tokenizer = build_model_and_tokenizer(cfg, device, args.model_type)
    model = load_checkpoint(model, checkpoint, device)

    print("=== Hate-speech demo ===")
    print(f"model: {args.model_type}")
    print(f"checkpoint: {checkpoint}")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        text = input("\nInput > ").strip()
        if text.lower() in ["quit", "exit", "q"]:
            break
        if not text:
            continue

        out = predict_text(
            text,
            model,
            tokenizer,
            device,
            threshold=args.threshold,
        )

        print("\n[Coarse]")
        print(f"  prediction: {out['coarse_pred']}")
        for i, p in enumerate(out["coarse_prob"]):
            print(f"    {COARSE_ID2LABEL[i]}: {p:.3f}")

        print("[Fine]")
        if out["fine_pred"]:
            print(f"  predicted labels: {', '.join(out['fine_pred'])}")
        else:
            print("  predicted labels: (none)")

        for i, p in enumerate(out["fine_prob"]):
            print(f"    {FINE_ID2LABEL[i]}: {p:.3f}")


if __name__ == "__main__":
    main()
