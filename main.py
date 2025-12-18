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
from fastapi.middleware.cors import CORSMiddleware
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
# 3. CORS 설정
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (프로덕션에서는 제한 필요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# 4. 텍스트 정규화
# ======================================================
def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[,\u00b7・]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======================================================
# 5. 강화된 의도 추출
# ======================================================
def extract_intent(text: str):
    """지역과 학과 계열/키워드를 추출"""

    # 지역 키워드
    regions = ["서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산",
               "강원", "충청", "전라", "경상", "제주"]

    # 학과 계열 키워드 (표준분류계열과 매칭)
    major_categories = {
        "인문": ["인문", "문학", "어학", "철학", "역사", "국어", "영어", "글쓰기", "번역"],
        "사회": ["사회", "경영", "경제", "법학", "행정", "정치", "기획"],
        "교육": ["교육", "유아교육", "교사", "상담"],
        "공학": ["공학", "컴퓨터", "소프트웨어", "전자", "기계", "화학", "건축",
                "프로그래밍", "코딩", "IT", "정보", "전기", "제작", "기술"],
        "자연": ["자연", "수학", "물리", "화학", "생물", "과학", "실험", "연구",
                "데이터", "분석", "통계"],
        "의약": ["의학", "약학", "간호", "의료", "건강", "보건"],
        "예체능": ["예술", "미술", "음악", "디자인", "체육", "창작", "예체능"]
    }

    # 지역 추출
    region = next((r for r in regions if r in text), None)

    # 학과 계열 추출
    detected_categories = []
    for category, keywords in major_categories.items():
        if any(keyword in text for keyword in keywords):
            detected_categories.append(category)

    # 모든 키워드 추출 (검색에 사용)
    all_keywords = []
    for keywords in major_categories.values():
        all_keywords.extend([k for k in keywords if k in text])

    return region, detected_categories, all_keywords


# ======================================================
# 6. 학과 DB + 임베딩 로드 (서버 시작 시 1회)
# ======================================================
print("📂 학과 벡터 DB 로딩 중...")

df = pd.read_csv("test_language.csv").fillna("")
df.columns = df.columns.str.strip()

# 🔥 컬럼명 오타 보정
df.rename(columns={"표준분 류계열(소)": "표준분류계열(소)"}, inplace=True)

# 🔥 지역 컬럼 통합 (핵심)
print("📋 CSV 컬럼 목록:")
print(df.columns.tolist())
print("\n🔍 소재지 관련 컬럼 찾기:")
location_columns = [col for col in df.columns if '소재지' in col or '지역' in col or '주소' in col]
print(f"   발견된 컬럼: {location_columns}")

if "소재지" in df.columns:
    df["지역"] = df["소재지"]
    print(f"   ✅ '소재지' 컬럼 사용")
elif "소재지(상세)" in df.columns:
    df["지역"] = df["소재지(상세)"]
    print(f"   ✅ '소재지(상세)' 컬럼 사용")
else:
    df["지역"] = ""
    print(f"   ⚠️  소재지 컬럼을 찾지 못했습니다! 지역 필터링이 작동하지 않습니다.")

# 🔥 임베딩 로드
df["embedding"] = df["embedding"].apply(
    lambda x: np.array(ast.literal_eval(x))
)
corpus_embeddings = np.vstack(df["embedding"].values)

# 🔥 임베딩 모델
model = SentenceTransformer("intfloat/multilingual-e5-base")

print("✅ 학과 DB 로딩 완료")


# ======================================================
# 7. 학과 검색 로직
# ======================================================
def search_major(user_query: str, top_k: int = 5):
    # 전역 df를 수정하지 않도록 먼저 복사
    df_work = df.copy()

    print(f"\n{'='*60}")
    print(f"🔍 검색 시작: {user_query[:100]}...")
    print(f"전체 학과 수: {len(df_work)}")

    # 임베딩 검색
    query = "query: " + normalize_text(user_query)
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(corpus_embeddings, query_embedding)
    df_work["score"] = scores

    # 의도 추출 (지역, 계열, 키워드)
    region, categories, keywords = extract_intent(user_query)

    print(f"📍 감지된 지역: {region}")
    print(f"📚 감지된 계열: {categories}")
    print(f"🔑 감지된 키워드: {keywords}")

    df_filtered = df_work

    # 🔥 지역 필터
    if region and "지역" in df_filtered.columns:
        before_count = len(df_filtered)

        # 지역 필터 적용 전 샘플 출력
        if before_count > 0:
            sample_regions = df_work["지역"].dropna().unique()[:20]
            print(f"📍 CSV의 실제 지역 값 샘플: {list(sample_regions)}")

        df_filtered = df_filtered[
            df_filtered["지역"]
            .astype(str)
            .str.contains(region, na=False)
        ]
        print(f"✅ 지역 필터 적용: {before_count} -> {len(df_filtered)} ('{region}' 포함)")

        # 매칭 안 된 경우 경고
        if len(df_filtered) == 0 and before_count > 0:
            print(f"⚠️  '{region}'과 매칭되는 지역이 없습니다!")

    # 🔥 표준계열 필터 (새로 추가!)
    if categories and "표준분류계열(중)" in df_filtered.columns:
        before_count = len(df_filtered)
        # 계열 중 하나라도 포함되면 선택
        category_filter = df_filtered["표준분류계열(중)"].apply(
            lambda x: any(cat in str(x) for cat in categories)
        )
        df_filtered = df_filtered[category_filter]
        print(f"✅ 계열 필터 적용: {before_count} -> {len(df_filtered)} (계열: {categories})")

        # 매칭 안 된 경우 샘플 출력
        if len(df_filtered) == 0 and before_count > 0:
            print(f"⚠️  CSV의 실제 계열 값 샘플:")
            sample_categories = df_work["표준분류계열(중)"].dropna().unique()[:10]
            print(f"   {list(sample_categories)}")

    # 🔥 키워드로 학과명/학과특성 필터 (추가 정확도 향상)
    if keywords:
        for keyword in keywords[:3]:  # 상위 3개 키워드만
            if "학과명" in df_filtered.columns:
                keyword_filter = (
                    df_filtered["학과명"].astype(str).str.contains(keyword, case=False, na=False) |
                    df_filtered["학과특성"].astype(str).str.contains(keyword, case=False, na=False)
                )
                # 키워드 매치된 항목에 점수 가산
                df_filtered.loc[keyword_filter, "score"] += 0.1

    print(f"📊 최종 결과: {len(df_filtered)}개 학과")
    print(f"{'='*60}\n")

    # 점수순 정렬 후 상위 반환
    return df_filtered.sort_values(
        "score",
        ascending=False
    ).head(top_k)


# ======================================================
# 8. GPT 프롬프트 생성
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
# 9. GPT 호출
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
# 10. API 요청 / 응답 모델
# ======================================================
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# ======================================================
# 11. 핵심 API 엔드포인트
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
