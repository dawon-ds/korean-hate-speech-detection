# interactive_infer.py
import torch
from transformers import AutoTokenizer

from src.config import load_config
from src.models import FlatHateModel, HierHateModel

# coarse / fine 라벨 이름 정의
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

# === 여기만 본인 경로로 고쳐서 사용 ===
CHECKPOINT_PATH = "checkpoints/flat_best.pt"  # 또는 hier_best.pt 등


def build_model_and_tokenizer(cfg, device):
    plm_name   = cfg.get("model", "plm_name")
    num_coarse = cfg.get("model", "num_coarse", default=3)
    num_fine   = cfg.get("model", "num_fine",   default=8)
    hierarchical = cfg.get("model", "hierarchical", default=False)

    lambda_fine = cfg.get("model", "lambda_fine", default=1.0)
    lambda_hier = cfg.get("model", "lambda_hier", default=1.0)

    if hierarchical:
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
    return model, tokenizer, hierarchical


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_text(text, model, tokenizer, device, threshold=0.5):
    # 토크나이즈
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
    logits_coarse = outputs["logits_coarse"]  # (1, 3)
    logits_fine   = outputs["logits_fine"]    # (1, 8)

    # coarse: softmax → argmax
    prob_coarse = torch.softmax(logits_coarse, dim=-1)[0].cpu().tolist()
    pred_coarse_id = int(torch.argmax(logits_coarse, dim=-1)[0].cpu().item())
    pred_coarse_label = COARSE_ID2LABEL[pred_coarse_id]

    # fine: sigmoid + threshold
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
    # 1) config / device
    cfg = load_config("config/base.yaml")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 2) 모델 / 토크나이저 로드
    model, tokenizer, hierarchical = build_model_and_tokenizer(cfg, device)
    model = load_checkpoint(model, CHECKPOINT_PATH, device)

    print("=== Hate-speech demo ===")
    print(f"model: {'Hier' if hierarchical else 'Flat'}")
    print("종료하려면 'quit', 'exit' 입력")

    while True:
        text = input("\n입력 문장 > ").strip()
        if text.lower() in ["quit", "exit", "q"]:
            break
        if not text:
            continue

        out = predict_text(text, model, tokenizer, device, threshold=0.5)

        # 결과 출력
        print("\n[Coarse]")
        print(f"  예측: {out['coarse_pred']}")
        for i, p in enumerate(out["coarse_prob"]):
            print(f"    {COARSE_ID2LABEL[i]}: {p:.3f}")

        print("[Fine]")
        if out["fine_pred"]:
            print(f"  예측 태그: {', '.join(out['fine_pred'])}")
        else:
            print("  예측 태그: (none)")

        for i, p in enumerate(out["fine_prob"]):
            print(f"    {FINE_ID2LABEL[i]}: {p:.3f}")


if __name__ == "__main__":
    main()
