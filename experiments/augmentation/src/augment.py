# src/augment.py

import os
import re
import json
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ===============================
# 0. PATH 설정
# ===============================
DATA_PATH = "data/merged_dataset_v1.1.csv"
AUTO_SEED_PATH = "data/auto_seed.json"              # 자동 seed 결과
MERGED_SEED_SAVE_PATH = "data/merged_seed_lexicon.json"
AUG_ONLY_SAVE_PATH = "data/merged_dataset_v1.1_obf_M3_only.csv"
AUG_FULL_SAVE_PATH = "data/merged_dataset_v1.1_obf_M3_full.csv"

# ===============================
# 1. 기존 수동 SEED_LEXICON
# ===============================
SEED_LEXICON = {
    # ---------------------- #
    # 1. 일반 욕설 (profanity)
    # ---------------------- #
    "시발": {
        "canonical": ["시발", "씨발", "씨발이", "시발이"],
        "jamo": ["ㅅㅂ", "ㅆㅂ"],
        "type": "profanity",
        "targets": [],
    },
    "씨팔": {
        "canonical": ["씨팔", "씨팔새끼"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "좆": {
        "canonical": ["좆", "좆같은", "좆같다", "좆같네", "좆같고", "좆같아서"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "새끼": {
        "canonical": ["새끼", "새끼들", "새끼들이", "새끼가"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "병신": {
        "canonical": ["병신", "병신들", "병신같은", "병신이고", "병신이라"],
        "jamo": ["ㅂㅅ", "ㅄ"],
        "type": "profanity",
        "targets": [],
    },
    "등신": {
        "canonical": ["등신"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "개새끼": {
        "canonical": ["개새끼", "개새끼들"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "씹새끼": {
        "canonical": ["씹새끼"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "개같은": {
        "canonical": ["개같은", "개같이"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "지랄": {
        "canonical": ["지랄하네", "지랄을"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },

    # ---------------------- #
    # 2. 성별/젠더 슬러 (gender)
    # ---------------------- #
    "한남": {
        "canonical": ["한남", "한남들", "한남은", "한남이", "한남새끼", "한남충", "한남충들"],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },
    "한녀": {
        "canonical": ["한녀", "한녀들", "한녀는", "한녀년", "한녀년들"],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },
    "김치녀": {
        "canonical": ["김치녀", "김치녀들", "김치녀같은"],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },
    "맘충": {
        "canonical": ["맘충"],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },
    "돼지년": {
        "canonical": ["돼지년"],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },
    "년": {
        "canonical": [
            "년들", "그년", "걸레년", "씨발년", "씨발년들", "씨발년이",
            "병신년", "병신년들", "김치년"
        ],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },

    # ---------------------- #
    # 3. LGBT / 성소수자 슬러
    # ---------------------- #
    "게이": {
        "canonical": ["게이", "게이들", "게이새끼", "게이같은"],
        "jamo": [],
        "type": "slur_LGBT",
        "targets": ["LGBT"],
    },
    "동성애자": {
        "canonical": ["동성애자", "동성애자들", "동성애자들이", "동성애는"],
        "jamo": [],
        "type": "slur_LGBT",
        "targets": ["LGBT"],
    },
    "후장": {
        "canonical": ["후장", "후장에"],
        "jamo": [],
        "type": "slur_LGBT",
        "targets": ["LGBT"],
    },
    "똥꼬충": {
        "canonical": ["똥꼬충"],
        "jamo": [],
        "type": "slur_LGBT",
        "targets": ["LGBT"],
    },

    # ---------------------- #
    # 4. 연령 (age)
    # ---------------------- #
    "틀딱": {
        "canonical": ["틀딱", "틀딱들", "틀니충", "틀니"],
        "jamo": [],
        "type": "slur_age",
        "targets": ["age"],
    },
    "급식충": {
        "canonical": ["급식충"],
        "jamo": [],
        "type": "slur_age",
        "targets": ["age"],
    },

    # ---------------------- #
    # 5. 지역 (region)
    # ---------------------- #
    "전라도": {
        "canonical": ["전라도"],
        "jamo": [],
        "type": "slur_region",
        "targets": ["region"],
    },

    # ---------------------- #
    # 6. 인종/민족/국적 (race)
    # ---------------------- #
    "짱깨": {
        "canonical": ["짱깨", "짱개"],
        "jamo": [],
        "type": "slur_race",
        "targets": ["race"],
    },
    "홍어": {
        "canonical": ["홍어", "홍어새끼"],
        "jamo": [],
        "type": "slur_race",
        "targets": ["race"],
    },
    "조센징": {
        "canonical": ["조센징", "조센"],
        "jamo": [],
        "type": "slur_race",
        "targets": ["race"],
    },
    "조선족": {
        "canonical": ["조선족"],
        "jamo": [],
        "type": "slur_race",
        "targets": ["race"],
    },
    "동남아": {
        "canonical": ["동남아", "똥남아"],
        "jamo": [],
        "type": "slur_race",
        "targets": ["race"],
    },

    # ---------------------- #
    # 7. 종교 (religion)
    # ---------------------- #
    "개독": {
        "canonical": ["개독", "개독이", "개독은"],
        "jamo": [],
        "type": "slur_religion",
        "targets": ["religion"],
    },
    "개슬람": {
        "canonical": ["개슬람", "개슬람새끼들"],
        "jamo": [],
        "type": "slur_religion",
        "targets": ["religion", "race"],
    },

    # ---------------------- #
    # 8. 계층/기타 (socioeconomic)
    # ---------------------- #
    "개돼지": {
        "canonical": ["개돼지"],
        "jamo": [],
        "type": "slur_socioeconomic",
        "targets": ["socioeconomic"],
    },
    "상폐녀": {
        "canonical": ["상폐녀", "상폐녀들", "상폐"],
        "jamo": [],
        "type": "slur_socioeconomic",
        "targets": ["socioeconomic"],
    },

    # ---------------------- #
    # 9. 일반 모욕
    # ---------------------- #
    "정신병자": {
        "canonical": ["정신병자", "정신병자들"],
        "jamo": [],
        "type": "abuse_general",
        "targets": [],
    },
    "버러지": {
        "canonical": ["버러지"],
        "jamo": [],
        "type": "abuse_general",
        "targets": [],
    },
    "꼬라지": {
        "canonical": ["꼬라지"],
        "jamo": [],
        "type": "abuse_general",
        "targets": [],
    },
    "역겹다": {
        "canonical": ["역겹다"],
        "jamo": [],
        "type": "abuse_general",
        "targets": [],
    },
}

# ===============================
# 2. auto_seed.json 정제용 설정
# ===============================
MIN_FREQ = 3  # auto_seed에서 최소 등장 횟수
YEAR_REGEX = re.compile(r"\d{4}년|\b(19[0-9]{2}|20[0-2][0-9])\b")

SAFE_WORDS = {
    "개념", "개인", "개발", "개성", "개방", "개인적", "개최", "개편", "개정",
    "개학", "개강", "개선", "개요"
}

BAN_PREFIX = ["개", "씨", "좆", "병신", "미친", "씹", "년", "새끼", "후장", "틀딱", "맘충", "급식충"]
SAFE_AFTER_GAE = {"념", "인", "발", "성", "방", "선", "편", "정", "학", "선", "정"}


def looks_like_insult(token: str) -> bool:
    """auto_seed 후보가 진짜 욕/혐오 느낌인지 heuristic으로 판별"""
    t = token.strip()

    if len(t) < 2:
        return False
    if t in SAFE_WORDS:
        return False
    if YEAR_REGEX.search(t):
        return False
    if t.isdigit():
        return False

    # 1) prefix 기반
    for p in BAN_PREFIX:
        if t.startswith(p):
            # '개'로 시작하지만 '개념', '개인' 같은 정상 단어는 제외
            if p == "개" and len(t) >= 2 and t[1] in SAFE_AFTER_GAE:
                return False
            return True

    # 2) 욕 패턴 fragment 기반
    FRAGMENTS = [
        "ㅅㅂ", "ㅄ", "ㅂㅅ", "좆", "병신", "씨발", "시발",
        "충", "년", "새끼", "게이", "동성애", "홍어", "짱깨",
        "조센징", "조선족", "동남아", "버러지", "정신병", "역겹", "꼬라지"
    ]
    if any(frag in t for frag in FRAGMENTS):
        # 숫자+년 형태는 제외
        if re.match(r"\d{4}년", t):
            return False
        return True

    return False


# ===============================
# 3. auto_seed.json 읽고 정제
# ===============================
def load_and_clean_auto_seed(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)  # { token: count }

    cleaned = {}
    for term, cnt in raw.items():
        term = term.strip()
        try:
            freq = int(cnt)
        except Exception:
            freq = 1

        if freq < MIN_FREQ:
            continue
        if not looks_like_insult(term):
            continue

        cleaned[term] = freq

    print(f"> auto_seed loaded: {len(raw)} → cleaned: {len(cleaned)}")
    return cleaned


# ===============================
# 4. SEED_LEXICON + auto_seed merge
# ===============================
def build_merged_seed_lexicon(seed_lexicon, auto_seed_clean):
    merged = dict(seed_lexicon)  # 얕은 복사

    # 기존 canonical/jamo 전체 모아두기
    existing_forms = set()
    for _, info in seed_lexicon.items():
        for c in info.get("canonical", []):
            existing_forms.add(c)
        for j in info.get("jamo", []):
            existing_forms.add(j)

    added = 0
    for term in auto_seed_clean.keys():
        if term in merged or term in existing_forms:
            continue

        merged[term] = {
            "canonical": [term],
            "jamo": [],
            "type": "auto_profanity",
            "targets": [],
        }
        added += 1

    print(f"> merged seed lexicon size: base={len(seed_lexicon)}, added_auto={added}, total={len(merged)}")
    return merged


# ===============================
# 5. Augmentation 유틸 (obfuscation)
# ===============================
SPECIAL_CHARS = ["#", "*", "^", "~", "!", "?"]
LAUGH = ["ㅋ", "ㅋㅋ", "ㅋㅋㅋ"]
NUM_MAP = {"이": "2", "일": "1", "팔": "8"}


def safe_insert_special(s: str) -> str:
    if len(s) <= 1:
        return s
    pos = random.randint(1, len(s) - 1)
    return s[:pos] + random.choice(SPECIAL_CHARS + LAUGH) + s[pos:]


def safe_spacing(s: str) -> str:
    if len(s) <= 2:
        return s
    pos = random.randint(1, len(s) - 1)
    return s[:pos] + " " + s[pos:]


def safe_elongate(s: str) -> str:
    if len(s) <= 1:
        return s
    idx = random.randint(0, len(s) - 1)
    return s[:idx + 1] + s[idx] * random.randint(1, 3) + s[idx + 1:]


def to_jamo_fake(s: str) -> str:
    # 간단한 예시만: 시발 → ㅅㅣㅂㅏㄹ류
    return s.replace("시발", "ㅅㅣㅂㅏㄹ").replace("씨발", "ㅆㅣㅂㅏㄹ")


def leet(s: str) -> str:
    for k, v in NUM_MAP.items():
        s = s.replace(k, v)
    return s


def random_obfuscate_token(token: str, seed_info: dict) -> str:
    if len(token) <= 1:
        return token

    rules = ["insert_special", "spacing", "elongate"]

    t = seed_info.get("type", "")
    if t == "profanity" or t.startswith("auto"):
        rules += ["to_jamo", "leet"]
    if t.startswith("slur"):
        rules += ["prefix_suffix"]

    n = random.randint(1, min(3, len(rules)))
    chosen = random.sample(rules, n)

    s = token
    for r in chosen:
        if r == "insert_special":
            s = safe_insert_special(s)
        elif r == "spacing":
            s = safe_spacing(s)
        elif r == "elongate":
            s = safe_elongate(s)
        elif r == "to_jamo":
            s = to_jamo_fake(s)
        elif r == "leet":
            s = leet(s)
        elif r == "prefix_suffix":
            if random.random() < 0.5:
                s = "개" + s
            else:
                s = s + "새끼"

    return s


# ===============================
# 6. 텍스트에서 seed 매칭 & 증강 (seed_lexicon 버전)
# ===============================
def find_seed_matches(text: str, seed_lexicon: dict):
    matches = []
    for seed, info in seed_lexicon.items():
        for form in info.get("canonical", []):
            if not form:
                continue
            for m in re.finditer(re.escape(form), text):
                matches.append((seed, info, m.start(), m.end()))
    return matches


def augment_text_with_seed(text: str, seed_lexicon: dict, max_new: int = 3):
    """
    실제 변형 로직이 들어있는 버전
    - seed_lexicon을 명시적으로 받음
    """
    matches = find_seed_matches(text, seed_lexicon)
    if not matches:
        return []

    new_texts = set()

    # 여러 번 돌리면서 랜덤 변형된 문장 최대 max_new개 얻기
    for _ in range(max_new * 3):
        s = text
        # 뒤에서부터 치환 → index shift 방지
        for _, info, start, end in sorted(matches, key=lambda x: x[2], reverse=True):
            ori = s[start:end]
            obf = random_obfuscate_token(ori, info)
            s = s[:start] + obf + s[end:]

        if s != text:
            new_texts.add(s)
        if len(new_texts) >= max_new:
            break

    return list(new_texts)


# ===============================
# 6-1. dataset.py에서 사용하는 wrapper
# ===============================
MERGED_SEED_PATH = Path("data/merged_seed_lexicon.json")
_GLOBAL_SEED_LEXICON = None


def get_seed_lexicon():
    """
    merged_seed_lexicon.json을 한 번만 읽어서 캐싱.
    파일이 없으면 수동 SEED_LEXICON만 사용.
    """
    global _GLOBAL_SEED_LEXICON

    if _GLOBAL_SEED_LEXICON is not None:
        return _GLOBAL_SEED_LEXICON

    if MERGED_SEED_PATH.exists():
        with open(MERGED_SEED_PATH, "r", encoding="utf-8") as f:
            _GLOBAL_SEED_LEXICON = json.load(f)
        print(f"[augment] loaded merged seed lexicon from {MERGED_SEED_PATH}")
    else:
        _GLOBAL_SEED_LEXICON = SEED_LEXICON
        print("[augment] merged_seed_lexicon.json not found, using manual SEED_LEXICON only.")

    return _GLOBAL_SEED_LEXICON


def augment_text(text: str, max_new: int = 3):
    """
    dataset.load_and_split에서 사용하는 최종 API
      new_texts = augment_text(text, max_new=...)
    내부에서 seed_lexicon을 자동 로드 후 augment_text_with_seed 호출.
    """
    seed_lexicon = get_seed_lexicon()
    return augment_text_with_seed(text, seed_lexicon, max_new=max_new)


# ===============================
# 7. 전체 데이터셋에 augmentation 적용 (오프라인용)
# ===============================
def run_augmentation(data_path: str, merged_seed: dict, max_new_per_sentence: int = 3):
    df = pd.read_csv(data_path)
    df["text"] = df["text"].astype(str)

    # hate / offensive 문장만 증강 대상으로
    if "hate_label" in df.columns:
        target_df = df[df["hate_label"].isin(["hate", "offensive"])].copy()
    else:
        target_df = df.copy()

    augmented_rows = []

    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Augmenting"):
        text = row["text"]
        new_texts = augment_text_with_seed(text, merged_seed, max_new=max_new_per_sentence)
        for nt in new_texts:
            new_row = row.copy()
            new_row["text"] = nt
            augmented_rows.append(new_row)

    aug_df = pd.DataFrame(augmented_rows)
    aug_df.to_csv(AUG_ONLY_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"\n> Augmented only dataset saved: {AUG_ONLY_SAVE_PATH}, count={len(aug_df)}")

    full_df = pd.concat([df, aug_df], ignore_index=True)
    full_df.to_csv(AUG_FULL_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"> Full (original + augmented) dataset saved: {AUG_FULL_SAVE_PATH}, total={len(full_df)}")

    return aug_df, full_df


# ===============================
# 8. MAIN (오프라인로 데이터 생성할 때만 사용)
# ===============================
def main():
    print(">> Loading auto_seed and cleaning...")
    auto_seed_clean = load_and_clean_auto_seed(AUTO_SEED_PATH)

    print(">> Merging SEED_LEXICON with auto_seed...")
    merged_seed = build_merged_seed_lexicon(SEED_LEXICON, auto_seed_clean)

    with open(MERGED_SEED_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_seed, f, ensure_ascii=False, indent=2)
    print(f"> merged seed lexicon saved: {MERGED_SEED_SAVE_PATH}")

    print(">> Running augmentation on dataset...")
    run_augmentation(DATA_PATH, merged_seed, max_new_per_sentence=3)


if __name__ == "__main__":
    main()
