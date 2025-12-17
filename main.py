# ======================================================
# 대학 · 학과 추천 API 서버
# FastAPI + Embedding + GPT-4o-mini
# ======================================================

import pandas as pd
import numpy as np
import ast
import re
import os

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv


# ======================================================
# 1. 환경 변수 로드 (.env)
# ======================================================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ======================================================
# 2. FastAPI 앱 생성
# ======================================================
app = FastAPI(
    title="대학 · 학과 추천 챗봇 API",
    description="Embedding + GPT 기반 대학 전공 추천",
    version="1.0.0"
)


# ======================================================
# 3. 텍스트 정규화
# ======================================================
def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[,\u00b7・]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======================================================
# 4. 간단한 의도 추출
# ======================================================
def extract_intent(text: str):
    regions = ["서울", "경기", "부산", "대구", "인천", "광주", "대전", "울산"]
    majors = ["컴퓨터", "소프트웨어", "AI", "인공지능", "정보", "데이터"]

    region = next((r for r in regions if r in text), None)
    major = next((m for m in majors if m.lower() in text.lower()), None)

    return region, major


# ======================================================
# 5. 학과 DB + 임베딩 로드 (서버 시작 시 1회)
# ======================================================
print("📂 학과 벡터 DB 로딩 중...")

df = pd.read_csv("test_language.csv").fillna("")
df.columns = df.columns.str.strip()

# 🔥 컬럼명 오타 보정
df.rename(columns={"표준분 류계열(소)": "표준분류계열(소)"}, inplace=True)

# 🔥 지역 컬럼 통합 (핵심)
if "소재지" in df.columns:
    df["지역"] = df["소재지"]
elif "소재지(상세)" in df.columns:
    df["지역"] = df["소재지(상세)"]
else:
    df["지역"] = ""

# 🔥 임베딩 로드
df["embedding"] = df["embedding"].apply(
    lambda x: np.array(ast.literal_eval(x))
)
corpus_embeddings = np.vstack(df["embedding"].values)

# 🔥 임베딩 모델
model = SentenceTransformer("intfloat/multilingual-e5-base")

print("✅ 학과 DB 로딩 완료")


# ======================================================
# 6. 학과 검색 로직
# ======================================================
def search_major(user_query: str, top_k: int = 3):
    query = "query: " + normalize_text(user_query)

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(corpus_embeddings, query_embedding)
    df["score"] = scores

    region, major = extract_intent(user_query)

    df_filtered = df.copy()

    # 🔥 지역 필터
    if region and "지역" in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered["지역"]
            .astype(str)
            .str.contains(region, na=False)
        ]

    # 🔥 학과명 필터
    if major and "학과명" in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered["학과명"]
            .astype(str)
            .str.contains(major, case=False, na=False)
        ]

    return df_filtered.sort_values(
        "score",
        ascending=False
    ).head(top_k)


# ======================================================
# 7. GPT 프롬프트 생성
# ======================================================
def build_gpt_prompt(user_query: str, results_df: pd.DataFrame):
    context = ""

    for _, row in results_df.iterrows():
        context += f"""
대학명: {row.get('대학명', '')}
학과명: {row.get('학과명', '')}
소재지: {row.get('지역', '')}
학과특성: {row.get('학과특성', '')}
표준계열: {row.get('표준분류계열(중)', '')}
---
"""

    return f"""
너는 한국의 대학 입시 및 진로 전문 상담 챗봇이다.

[사용자 질문]
{user_query}

[추천 가능한 학과 정보]
{context}

요청사항:
1. 질문에 맞는 학과를 추천해라
2. 각 학과의 특징을 쉽게 설명해라
3. 졸업 후 진로와 전망을 설명해라
4. 친절한 한국어로 답변해라
5. 제공된 정보 외의 내용은 지어내지 마라
"""


# ======================================================
# 8. GPT 호출
# ======================================================
def call_gpt4_mini(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 대학 입시 전문 상담 챗봇이다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content


# ======================================================
# 9. API 요청 / 응답 모델
# ======================================================
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# ======================================================
# 10. 핵심 API 엔드포인트
# ======================================================
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    results = search_major(req.question)

    if results.empty:
        return ChatResponse(
            answer="조건에 맞는 학과를 찾지 못했어요. 질문을 조금 바꿔보세요."
        )

    prompt = build_gpt_prompt(req.question, results)
    answer = call_gpt4_mini(prompt)

    return ChatResponse(answer=answer)
