import pandas as pd

# -------------------------
# 파일 경로
# -------------------------
train_tsv_path = "./raw_data/unsmile_train_v1.0.tsv"
valid_tsv_path = "./raw_data/unsmile_valid_v1.0.tsv"

train_csv_path = "./data/unsmile_train.csv"
valid_csv_path = "./data/unsmile_valid.csv"

# -------------------------
# TSV 읽기
# -------------------------
train_df = pd.read_csv(train_tsv_path, sep="\t", encoding="utf-8")
valid_df = pd.read_csv(valid_tsv_path, sep="\t", encoding="utf-8")

# -------------------------
# CSV 저장
# -------------------------
train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
valid_df.to_csv(valid_csv_path, index=False, encoding="utf-8-sig")

print("Converted:")
print(" -", train_csv_path)
print(" -", valid_csv_path)
