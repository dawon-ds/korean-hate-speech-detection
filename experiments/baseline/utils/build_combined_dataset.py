import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# -------------------------------------------------
# 1) HateScore 로드 및 UnSmile 스키마로 매핑
# -------------------------------------------------
hatescore_path = DATA_DIR / "hatescore.csv"  # macrolabel, microlabel, comment 포함된 CSV
print("[INFO] Load HateScore:", hatescore_path)
hatescore = pd.read_csv(hatescore_path, encoding="utf-8-sig")

print(hatescore.head(3))
print("[INFO] macrolabel 분포:")
print(hatescore["macrolabel"].value_counts())
print("[INFO] microlabel 분포:")
print(hatescore["microlabel"].value_counts())

# 최종 사용할 레이블 컬럼 (UnSmile 스키마)
label_cols = [
    "여성/가족", "남성", "성소수자", "인종/국적",
    "연령", "지역", "종교", "기타혐오",
    "악플/욕설", "clean",
]

# 새 데이터프레임 뼈대
hs_proc = pd.DataFrame()
hs_proc["문장"] = hatescore["comment"]
for c in label_cols:
    hs_proc[c] = 0


def fill_labels(row):
    macro = row["macrolabel"]
    micro = row["microlabel"]

    labels = {
        "여성/가족": 0,
        "남성": 0,
        "성소수자": 0,
        "인종/국적": 0,
        "연령": 0,
        "지역": 0,
        "종교": 0,
        "기타혐오": 0,
        "악플/욕설": 0,
        "clean": 0,
    }

    # 1) 혐오발언인 경우: microlabel에 따라 타겟 선택
    if macro == "혐오발언":
        if micro in ["여성", "여성/가족"]:
            labels["여성/가족"] = 1
        elif micro == "남성":
            labels["남성"] = 1
        elif micro == "성소수자":
            labels["성소수자"] = 1
        elif micro == "인종/국적":
            labels["인종/국적"] = 1
        elif micro == "연령":
            labels["연령"] = 1
        elif micro == "지역":
            labels["지역"] = 1
        elif micro == "종교":
            labels["종교"] = 1
        elif micro in ["기타혐오", "기타", "기타 혐오"]:
            labels["기타혐오"] = 1

    # 2) 단순 악플
    elif macro in ["단순악플", "단순 악플"]:
        labels["악플/욕설"] = 1

    # 3) 일반/중립/클린 계열
    elif macro in ["중립", "일반 댓글", "일반", "clean", "None"]:
        labels["clean"] = 1

    # 그 외 애매한 macro 값은 일단 clean=0, 나머지도 0으로 둠
    return pd.Series(labels)


hs_labels = hatescore.apply(fill_labels, axis=1)
for c in label_cols:
    hs_proc[c] = hs_labels[c]

hs_proc["source"] = "hatescore"

print("[INFO] HateScore 처리 예시:")
print(hs_proc.head(3))

# -------------------------------------------------
# 2) UnSmile train/valid CSV 로드
# -------------------------------------------------
unsmile_train_path = DATA_DIR / "unsmile_train.csv"
unsmile_valid_path = DATA_DIR / "unsmile_valid.csv"

print("[INFO] Load UnSmile train:", unsmile_train_path)
unsmile_train = pd.read_csv(unsmile_train_path, encoding="utf-8-sig")

print("[INFO] Load UnSmile valid (test용):", unsmile_valid_path)
unsmile_valid = pd.read_csv(unsmile_valid_path, encoding="utf-8-sig")

# 컬럼 이름 통일 (기타 혐오 -> 기타혐오)
if "기타 혐오" in unsmile_train.columns:
    unsmile_train = unsmile_train.rename(columns={"기타 혐오": "기타혐오"})
if "기타 혐오" in unsmile_valid.columns:
    unsmile_valid = unsmile_valid.rename(columns={"기타 혐오": "기타혐오"})

# 혹시 UnSmile 쪽에 label_cols 중 빠진 게 있으면 0으로 채우기
for c in label_cols:
    if c not in unsmile_train.columns:
        print(f"[WARN] UnSmile train: '{c}' 컬럼이 없어 0으로 추가합니다.")
        unsmile_train[c] = 0

# 최종 사용할 컬럼만 추출 (train만)
unsmile_train_proc = unsmile_train[["문장"] + label_cols].copy()
unsmile_train_proc["source"] = "unsmile_train"

print("[INFO] UnSmile train 처리 예시:")
print(unsmile_train_proc.head(3))

# -------------------------------------------------
# 3) 최종 train = UnSmile train + HateScore
#    test = UnSmile valid (이 스크립트에서는 저장만, split/work는 experiments에서)
# -------------------------------------------------
combined_train = pd.concat([unsmile_train_proc, hs_proc], ignore_index=True)
print("[INFO] Combined train source 분포:")
print(combined_train["source"].value_counts())

train_out_path = DATA_DIR / "train_hatescore_unsmile.csv"
combined_train.to_csv(train_out_path, index=False, encoding="utf-8-sig")
print("[INFO] Saved combined train ->", train_out_path)

print("[INFO] Done.")
