# import pandas as pd

# # =========================================
# # 1. 원본 CSV 로드
# # =========================================
# INPUT_CSV = "score.csv"

# df = pd.read_csv(INPUT_CSV).fillna("")

# # =========================================
# # 2. 임베딩용 컬럼
# # =========================================
# embedding_cols = [
#     "대학명",
#     "단과대학",
#     "학과명",
#     "소재지",
#     "소재지(상세)",
#     "주야구분",
#     "학과특성",
#     "표준분류계열(대)",
#     "표준분류계열(중)",
#     "표준분류계열(소)",
# ]

# # =========================================
# # 3. 성적용 컬럼
# # =========================================
# score_cols = [
#     "대학명",
#     "학과명",
#     "학점",
#     "모집인원",
#     "경쟁률",
# ]

# # =========================================
# # 4. CSV 분할
# # =========================================
# df_embedding = df[embedding_cols].copy()
# df_score = df[score_cols].copy()

# # =========================================
# # 5. 저장
# # =========================================
# df_embedding.to_csv("A_embedding.csv", index=False)
# df_score.to_csv("A_score.csv", index=False)

# print("✅ CSV 분할 완료")
# print(" - A_embedding.csv (임베딩용)")
# print(" - A_score.csv (성적용)")

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# =========================================
# 1. 텍스트 정규화
# =========================================
def normalize_text(text):
    text = str(text)
    text = re.sub(r"[,\u00b7・]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================================
# 2. 임베딩 문장 생성
# =========================================
def make_passage(row):
    fields = [
        row["대학명"],
        row["단과대학"],
        row["학과명"],
        row["소재지"],
        row["소재지(상세)"],
        row["주야구분"],
        row["학과특성"],
        row["표준분류계열(중)"],
    ]

    fields = [normalize_text(f) for f in fields if f]
    return "passage: " + " ".join(fields)

# =========================================
# 3. CSV 로드
# =========================================
INPUT_CSV = "A_embedding.csv"
df = pd.read_csv(INPUT_CSV).fillna("")

# =========================================
# 4. 임베딩 텍스트 생성
# =========================================
df["embedding_text"] = df.apply(make_passage, axis=1)

# =========================================
# 5. 모델 로드
# =========================================
model = SentenceTransformer("intfloat/multilingual-e5-base")

# =========================================
# 6. 임베딩 생성
# =========================================
embeddings = model.encode(
    df["embedding_text"].tolist(),
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# =========================================
# 7. 저장
# =========================================
out_df = pd.DataFrame({
    "embedding": embeddings.tolist()
})

out_df.to_csv("A_embedding_vectors.csv", index=False)

print("✅ 임베딩 생성 완료")
print(" - A_embedding_vectors.csv")
