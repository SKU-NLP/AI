import pandas as pd
import numpy as np
import ast
import re
import os

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# ======================================================
# 0. 기본 설정
# ======================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="맥락 + 성적 + 학교평점 기반 대학 추천 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 1. 텍스트 정규화
# ======================================================
def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[,\u00b7・]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ======================================================
# 2. 사용자 질문에서 내신 추출
# ======================================================
def extract_grade(text: str):
    match = re.search(r"(?:내신\s*)?(\d(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None

# ======================================================
# 3. 데이터 로딩
# ======================================================
print("📂 데이터 로딩 중...")

# 학과 정보
info_df = pd.read_csv("A_embedding.csv").fillna("")
info_df.columns = info_df.columns.str.strip()

# 임베딩
emb_df = pd.read_csv("A_embedding_vectors.csv")
emb_df["embedding"] = emb_df["embedding"].apply(
    lambda x: np.array(ast.literal_eval(x))
)

# 학과 평균 내신
score_df = pd.read_csv("A_score.csv").fillna("")
score_df.columns = score_df.columns.str.strip()

# 🔥 학교 평점
school_df = pd.read_csv("school_score.csv").fillna("")
school_df.columns = (
    school_df.columns
    .str.replace("\n", "", regex=False)
    .str.replace(" ", "", regex=False)
)

school_df = school_df.rename(columns={
    "학문적평판점수학계에서얼마나인정받는대학인가": "학문적평판점수",
    "취업평판점수기업이선호나는학교인가": "취업평판점수",
    "교육밀도교수수대비학생수.학생한명이교수에게얼마나집중지도를받을수있는가": "교육밀도",
    "교수당논문인용수교수의질.": "교수당논문인용수",
    "국제공동연구유학해외연구글로벌진출연결성": "국제공동연구",
    "졸업생성과취업률졸업생에": "졸업생성과",
})

# 병합
info_df["embedding"] = emb_df["embedding"].values
corpus_embeddings = np.vstack(info_df["embedding"].values)

# 지역
if "소재지(상세)" in info_df.columns:
    info_df["지역"] = info_df["소재지(상세)"]
else:
    info_df["지역"] = info_df.get("소재지", "")

model = SentenceTransformer("intfloat/multilingual-e5-base")

print("✅ 데이터 로딩 완료")

# ======================================================
# 4. 임베딩 기반 학과 검색
# ======================================================
def search_major_contextual(user_query: str, top_k: int = 8):
    query_emb = model.encode(
        "query: " + normalize_text(user_query),
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    df = info_df.copy()
    df["sim"] = np.dot(corpus_embeddings, query_emb)
    results = df.sort_values("sim", ascending=False).head(top_k)

    if results.empty or results.iloc[0]["sim"] < 0.65:
        return pd.DataFrame()

    return results

# ======================================================
# 5. 성적 + 학교 평점 매칭
# ======================================================
def attach_score(results_df: pd.DataFrame, user_grade: float):
    rows = []

    for _, row in results_df.iterrows():
        score_row = score_df[
            (score_df["대학명"] == row["대학명"]) &
            (score_df["학과명"] == row["학과명"])
        ]
        if score_row.empty:
            continue

        try:
            avg = float(score_row.iloc[0]["학점"])
        except:
            continue

        # 🔥 내신 낮을수록 유리
        if user_grade >= avg + 0.2:
            level = "상향"
        elif user_grade <= avg - 0.2:
            level = "하향"
        else:
            level = "적정"

        school_row = school_df[school_df["학교"] == row["대학명"]]
        school = school_row.iloc[0] if not school_row.empty else {}

        rows.append({
            "대학명": row["대학명"],
            "학과명": row["학과명"],
            "지역": row["지역"],
            "평균내신": avg,
            "판단": level,

            "학문평판": school.get("학문적평판점수"),
            "취업평판": school.get("취업평판점수"),
            "교육밀도": school.get("교육밀도"),
            "교수연구력": school.get("교수당논문인용수"),
            "국제화": school.get("국제공동연구"),
            "졸업성과": school.get("졸업생성과"),
            "학교순위": school.get("순위"),
        })

    return pd.DataFrame(rows)

# ======================================================
# 6. GPT 프롬프트
# ======================================================
def build_hybrid_prompt(user_query, user_grade, rec_df):
    context = ""

    for idx, r in rec_df.iterrows():
        context += f"""
[{idx+1}] {r['대학명']} {r['학과명']} ({r['판단']})
- 평균 내신: {r['평균내신']}
- 학교 평가
  · 학문적 평판 점수: {r['학문평판']}
  · 취업 평판 점수: {r['취업평판']}
  · 교육 밀도: {r['교육밀도']}
  · 교수 연구력: {r['교수연구력']}
  · 국제화 수준: {r['국제화']}
  · 졸업 성과: {r['졸업성과']}
"""
    return f"""
너는 한국 대학 입시 전문 상담 챗봇이다.

[사용자 정보]
- 질문: {user_query}
- 내신: {user_grade}

[추천 결과]
아래 모든 추천 항목에 대해
① 내신 적합도
② 학교 평가 지표
③ 왜 이 학생에게 적합한지
를 **반드시 모두 포함해서 설명해라.**

{context if context else "조건에 맞는 추천이 부족함"}

[출력 규칙 — 매우 중요]
- 모든 추천 번호마다 반드시 **학교 평가 항목을 전부 설명**
- 절대로 첫 번째만 자세히 설명하고 나머지를 생략하지 마라
- 각 학교를 비교·분석하는 말투로 설명해라
- 상담 선생님처럼 자연스럽게 말해라
"""

# ======================================================
# 7. API
# ======================================================
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    user_grade = extract_grade(req.question)
    majors = search_major_contextual(req.question)

    rec_df = (
        attach_score(majors, user_grade)
        if user_grade is not None and not majors.empty
        else pd.DataFrame()
    )

    prompt = build_hybrid_prompt(req.question, user_grade, rec_df)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 입시 상담 전문가다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return {
        "answer": response.choices[0].message.content,
        "matched_count": len(rec_df)
    }
