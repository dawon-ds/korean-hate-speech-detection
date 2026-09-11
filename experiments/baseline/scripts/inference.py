import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_A_PATH = "./model/modelA_name"
MODEL_B_PATH = "./model/modelB_name"

TEST_SENTENCES = [
    "치료는 하나님한테 부탁해라",
    "교회얘기 지긋지긋하다...",
    "꼬우면 돌아가던가ㅋ",
]

THRESHOLD = 0.30


ID2LABEL_DEFAULT = {
    0: "clean",
    1: "악플/욕설",
    2: "여성/가족",
    3: "남성",
    4: "성소수자",
    5: "인종/국적",
    6: "연령",
    7: "지역",
    8: "종교",
    9: "기타혐오",
}


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if hasattr(model.config, "id2label") and model.config.id2label:
        raw = model.config.id2label
        id2label = {int(k): v for k, v in raw.items()}
    else:
        id2label = ID2LABEL_DEFAULT

    return tokenizer, model, device, id2label



@torch.no_grad()
def infer_with_threshold(texts, tokenizer, model, device, id2label, threshold: float):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = model(**encoded)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    results = []
    for i, text in enumerate(texts):
        p = probs[i]
        pred_id = int(p.argmax())
        pred_label = id2label[pred_id]

        suspected = [id2label[j] for j in range(len(p)) if p[j] >= threshold]
        if not suspected:
            suspected = ["none"]

        scores = {id2label[j]: float(p[j]) for j in range(len(p))}

        results.append(
            {
                "text": text,
                "pred_label": pred_label,
                "pred_id": pred_id,
                "suspected": suspected,
                "scores": scores,
            }
        )

    return results

def main():
    print(f"[INFO] 모델 A 로딩: {MODEL_A_PATH}")
    tokA, modA, devA, id2labelA = load_model(MODEL_A_PATH)

    print(f"[INFO] 모델 B 로딩: {MODEL_B_PATH}")
    tokB, modB, devB, id2labelB = load_model(MODEL_B_PATH)

    print("\n=========== 테스트 문장 ===========")
    for s in TEST_SENTENCES:
        print(" -", s)

    resA = infer_with_threshold(TEST_SENTENCES, tokA, modA, devA, id2labelA, THRESHOLD)
    resB = infer_with_threshold(TEST_SENTENCES, tokB, modB, devB, id2labelB, THRESHOLD)

    print("\n" + "=" * 70)
    print("두 모델 비교 (softmax + threshold)")
    print("=" * 70)

    for i, text in enumerate(TEST_SENTENCES):
        print("\n--------------------------------------------------------")
        print(f"[문장] {text}\n")

        # 모델 A
        print(f"Unsmile: ")
        print(f" - 최종 라벨: {resA[i]['pred_label']} (id={resA[i]['pred_id']})")
        print(f" - 의심 라벨: {', '.join(resA[i]['suspected'])}")
        print("라벨별 확률:")
        for label, score in resA[i]["scores"].items():
            print(f"{label:>8s}: {score:.4f}")

        print()

        # 모델 B
        print(f"Unsmile+HateScore: ")
        print(f" - 최종 라벨: {resB[i]['pred_label']} (id={resB[i]['pred_id']})")
        print(f" - 의심 라벨: : {', '.join(resB[i]['suspected'])}")
        print("라벨별 확률:")
        for label, score in resB[i]["scores"].items():
            print(f"{label:>8s}: {score:.4f}")


if __name__ == "__main__":
    main()
