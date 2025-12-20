# main.py
# 실행: uvicorn main:app --reload
# 필요 패키지:
# pip install fastapi uvicorn python-dotenv openai langchain langchain-openai langchain-core sentence-transformers pandas numpy

import os
import json
import re
import ast
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from recommendation_models import RecommendationSummaryRequest, RecommendationSummaryResponse


# ======================================================
# 0) 기본 설정
# ======================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 확인하세요.")

# OpenAI 공식 SDK(필요하면 사용)
client = OpenAI(api_key=OPENAI_API_KEY)

# LangChain LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=512,
)

app = FastAPI(title="맥락 + 성적 + 학교평점 기반 대학 추천 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# 1) 세션 메모리 (서버 인메모리)
# - 실제 서비스면 Redis/DB로 교체 권장
# ======================================================
conversation_store: dict[str, list[dict]] = {}
MAX_TURNS = 12  # 최근 12턴 정도만 유지(너무 길어지면 비용/토큰 증가)

def get_history(session_id: str) -> list[dict]:
    return conversation_store.get(session_id, [])

def append_history(session_id: str, role: str, content: str):
    conversation_store.setdefault(session_id, []).append({"role": role, "content": content})
    # 길이 제한
    if len(conversation_store[session_id]) > MAX_TURNS * 2:
        conversation_store[session_id] = conversation_store[session_id][-MAX_TURNS * 2 :]

def history_to_text(history: list[dict]) -> str:
    if not history:
        return "없음"
    lines = []
    for m in history:
        r = "사용자" if m["role"] == "user" else "상담봇"
        lines.append(f"{r}: {m['content']}")
    return "\n".join(lines)


# ======================================================
# 2) 유틸
# ======================================================
def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[,\u00b7・]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_grade(text: str):
    # "내신 2.7", "2.7" 같은 케이스 대응(원하는 규칙 더 강화 가능)
    match = re.search(r"(?:내신\s*)?(\d(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None

def safe_float(x):
    try:
        return float(x)
    except:
        return None


# ======================================================
# 3) LangChain 체인들 (Intent / Answer)
# ======================================================
def build_chain(template: str):
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()


# 선택: 의도 분류(멀티체인까지는 아니고, 후처리 분기 정도에 사용)
INTENT_PROMPT = """
너는 사용자의 질문 의도를 분류한다.
아래 라벨 중 하나만 정확히 출력해라(다른 말 금지).

라벨:
- RECOMMEND: 조건에 맞는 대학/학과를 추천해달라는 요청
- SCHOOL_INFO: 특정 학교/학과 하나에 대한 정보 요청(예: 성균관대 컴공 어때?)
- EXPLAIN_SYSTEM: 시스템/코드/구조 설명 요청
- OTHER: 그 외

질문:
{question}

라벨:
"""

# 최종 답변 프롬프트(추천/단일학교/설명 모두 커버)
ANSWER_PROMPT = """
너는 입시 데이터를 기반으로 냉철하고 현실적인 조언을 해주는 '진학 담당 선생님'이다.
학생의 인생이 달린 문제다. 희망 고문 하지 말고 데이터에 근거해 솔직하게 말해라.

[필수 규칙]
1. **중복 금지**: 절대 같은 학교/학과 설명을 두 번 반복하지 마라.
2. **숫자 나열 금지**: "학문평판=68.4" 처럼 기계적으로 말하지 말고, "학문적 성과가 매우 우수하며(68.4점)..." 처럼 문장에 자연스럽게 녹여라.
3. **현실적 조언**: 
   - 학생 내신이 평균 내신보다 0.5등급 이상 낮으면 "상향 지원"임을 분명히 하고, 합격이 어려울 수 있음을 경고해라.
   - 무조건 "적합하다"고 하지 말고, 성적이 부족하면 성적 리스크를 먼저 언급해라.

[입력 정보]
- 이전 대화: {history}
- 사용자 질문: {question}
- 의도: {intent}
- 검색된 대학 데이터: 
{context}

[답변 가이드]
1. **intent가 RECOMMEND(추천)일 때**:
   - 검색된 후보 중 가장 적합한 3개 정도만 추려서 보여줘.
   - 각 대학별로:
     (1) [대학명 / 학과명] (지역)
     (2) 내신 분석: 학생 내신 vs 평균 내신 비교. (안전/적정/상향/위험 판정)
     (3) 핵심 강점: 학교 평판, 취업률 등 지표를 활용해 이 학교의 장점을 서술형으로 요약.
     (4) 선생님의 한마디: 왜 이 학생에게 추천하는지, 주의할 점은 무엇인지.

2. **intent가 SCHOOL_INFO(상세정보)일 때**:
   - 사용자가 물어본 **딱 그 학교/학과 하나**에 대해서만 집중적으로 분석해라. (다른 학교 언급 X)
   - 지표 점수를 나열하지 말고, 그 점수가 의미하는 바(예: 연구력이 좋다, 취업이 잘 된다)를 설명해라.
   - **가장 중요한 건 내신 적합성이다.** 성적 차이가 크면 냉정하게 말해줘라.

3. 답변은 친절하지만 단호한 '해요체'를 사용해라.
4. 가독성을 위해 불필요한 줄바꿈이나 기호를 남발하지 마라.
"""

FALLBACK_PROMPT = """
너는 대학·학과 진학 상담을 돕는 챗봇이다.

중요: 지금은 내부 RAG(학과/평판/내신 데이터)에서 충분한 근거를 찾지 못했다.
그러므로 특정 대학/학과의 '평균 내신'이나 '평판 점수' 같은 수치를 지어내면 안 된다.

할 일:
1) 사용자의 질문에 대해 일반적으로 알려진 정보 수준에서 설명한다.
2) 정확한 추천을 위해 필요한 추가 질문 2~4개를 제시한다.
3) 답변은 친절한 해요체로, 과장 없이 말한다.
4) 모르면 모른다고 말하고, 확인 방법(예: 학교 입학처/학과 커리큘럼/취업통계 확인)을 안내한다.

이전 대화:
{history}

사용자 질문:
{question}

답변:
"""

fallback_chain = build_chain(FALLBACK_PROMPT)
intent_chain = build_chain(INTENT_PROMPT)
answer_chain = build_chain(ANSWER_PROMPT)
# 추천 요약 API에서 사용할 LLM 프롬프트 템플릿이다.
# 프론트가 필요한 JSON 스키마를 정확히 맞추도록 강제하기 위해
# 입력 필드와 출력 스키마를 명시하고, JSON 외 텍스트 출력 금지 규칙을 포함한다.
SUMMARY_PROMPT = """
너는 대학/학과 추천 시스템의 요약 결과를 JSON으로 생성한다.
반드시 아래 스키마를 지켜서 JSON만 출력해라. 마크다운/설명 금지.

[입력]
- 이전 대화: {history}
- 사용자 질문: {question}
- 추천 후보 데이터:
{context}

[출력 JSON 스키마]
{{
  "departments": [
    {{
      "name": "학과명",
      "university": "대학명",
      "matchScore": 0-100 정수,
      "employmentRate": 0-100 정수,
      "averageSalary": 만원 단위 정수,
      "competitionRate": 실수,
      "requiredGrade": "예: 2.3등급",
      "description": "추천 이유 요약",
      "relatedJobs": ["직업1", "직업2"],
      "websiteUrl": "https://..."
    }}
  ],
  "userProfile": {{
    "interests": ["관심사1", "관심사2"],
    "strengths": [
      {{ "subject": "과목", "score": 0-100 정수 }}
    ],
    "careerGoal": "진로 목표",
    "region": "희망 지역",
    "gradeLevel": "고1/고2/고3/졸업"
  }},
  "initialConversations": [
    {{ "user": "질문", "assistant": "응답" }}
  ],
  "afterRecommendationConversations": [
    {{ "user": "질문", "assistant": "응답" }}
  ],
  "industryTrends": [
    {{ "year": "2020", "demand": 0-100 정수, "salary": 만원 단위 정수 }}
  ],
  "skillRequirements": [
    {{ "skill": "역량", "importance": 0-100 정수 }}
  ]
}}

[규칙]
- JSON 외 텍스트 출력 금지.
- 값이 불확실하면 추정하되, 질문/후보 데이터와 일관되게 작성.
"""
# 요약용 프롬프트를 LLM 체인으로 구성한다.
# 요약 API는 이 체인을 통해 JSON 형태의 결과를 생성한다.
summary_chain = build_chain(SUMMARY_PROMPT)


# ======================================================
# 4) 데이터 로딩
# ======================================================
print("📂 데이터 로딩 중...")

info_df = pd.read_csv("A_embedding.csv").fillna("")
info_df.columns = info_df.columns.str.strip()

emb_df = pd.read_csv("A_embedding_vectors.csv").fillna("")
emb_df["embedding"] = emb_df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))

score_df = pd.read_csv("A_score.csv").fillna("")
score_df.columns = score_df.columns.str.strip()

school_df = pd.read_csv("school_score.csv").fillna("")
school_df.columns = (
    school_df.columns.str.replace("\n", "", regex=False).str.replace(" ", "", regex=False)
)

school_df = school_df.rename(columns={
    "학문적평판점수학계에서얼마나인정받는대학인가": "학문적평판점수",
    "취업평판점수기업이선호나는학교인가": "취업평판점수",
    "교육밀도교수수대비학생수.학생한명이교수에게얼마나집중지도를받을수있는가": "교육밀도",
    "교수당논문인용수교수의질.": "교수당논문인용수",
    "국제공동연구유학해외연구글로벌진출연결성": "국제공동연구",
    "졸업생성과취업률졸업생에": "졸업생성과",
})

# 임베딩 병합
info_df["embedding"] = emb_df["embedding"].values
corpus_embeddings = np.vstack(info_df["embedding"].values)

# 지역 컬럼 통일
if "소재지(상세)" in info_df.columns:
    info_df["지역"] = info_df["소재지(상세)"]
else:
    info_df["지역"] = info_df.get("소재지", "")

# SentenceTransformer(로컬 임베딩)
model = SentenceTransformer("intfloat/multilingual-e5-base")

print("✅ 데이터 로딩 완료")


# ======================================================
# 5) 검색 / 평가 
# ======================================================
def search_major_contextual(user_query: str, top_k: int = 20) -> pd.DataFrame:
    query_emb = model.encode(
        "query: " + normalize_text(user_query),
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    df = info_df.copy()
    df["sim"] = np.dot(corpus_embeddings, query_emb)
    return df.sort_values("sim", ascending=False).head(top_k)

def evaluate_retrieval(results_df: pd.DataFrame) -> bool:
    # 네 기존 기준 유지(원하면 튜닝)
    if results_df is None or results_df.empty:
        return False
    if float(results_df.iloc[0]["sim"]) < 0.80:
        return False
    if len(results_df) < 3:
        return False
    return True

def rerank_with_filters(results_df: pd.DataFrame, user_query: str) -> pd.DataFrame:
    """
    간단 리랭킹:
    - 쿼리 키워드(지역/주야/학과특성/계열)가 포함되면 가산점
    - 완전한 LLM rerank가 아니라 규칙 기반(가볍고 빠름)
    """
    q = normalize_text(user_query)
    tokens = set(q.split())

    def bonus(row):
        text = " ".join([
            normalize_text(row.get("대학명", "")),
            normalize_text(row.get("학과명", "")),
            normalize_text(row.get("지역", "")),
            normalize_text(row.get("주야구분", "")),
            normalize_text(row.get("학과특성", "")),
            normalize_text(row.get("표준분류계열(중)", "")),
        ])
        hit = sum(1 for t in tokens if t and t in text)
        return min(hit, 10) * 0.01  # 최대 +0.10

    df = results_df.copy()
    df["bonus"] = df.apply(bonus, axis=1)
    df["score"] = df["sim"] + df["bonus"]
    return df.sort_values("score", ascending=False)

def attach_score(results_df: pd.DataFrame, user_grade: float) -> pd.DataFrame:
    rows = []
    for _, row in results_df.iterrows():
        score_row = score_df[
            (score_df["대학명"] == row["대학명"]) &
            (score_df["학과명"] == row["학과명"])
        ]
        if score_row.empty:
            continue

        avg = safe_float(score_row.iloc[0].get("학점"))
        if avg is None or user_grade is None:
            level = "내신 판단 불가"
        else:
            if user_grade >= avg + 0.2:
                level = "하향"
            elif user_grade <= avg - 0.2:
                level = "상향"
            else:
                level = "적정"

        school_row = school_df[school_df.get("학교") == row["대학명"]] if "학교" in school_df.columns else pd.DataFrame()
        school = school_row.iloc[0] if not school_row.empty else {}

        rows.append({
            "대학명": row.get("대학명"),
            "학과명": row.get("학과명"),
            "지역": row.get("지역"),
            "주야구분": row.get("주야구분", ""),
            "학과특성": row.get("학과특성", ""),
            "계열": row.get("표준분류계열(중)", ""),
            "유사도": float(row.get("sim", 0.0)),
            "보너스": float(row.get("bonus", 0.0)),
            "최종점수": float(row.get("score", 0.0)),

            "평균내신": avg,
            "판단": level,

            "학문평판": school.get("학문적평판점수", "데이터 없음"),
            "취업평판": school.get("취업평판점수", "데이터 없음"),
            "교육밀도": school.get("교육밀도", "데이터 없음"),
            "교수연구력": school.get("교수당논문인용수", "데이터 없음"),
            "국제화": school.get("국제공동연구", "데이터 없음"),
            "졸업성과": school.get("졸업생성과", "데이터 없음"),
            "학교순위": school.get("순위", "데이터 없음"),
        })

    return pd.DataFrame(rows)


def build_context_text(rec_df: pd.DataFrame, top_n: int = 8) -> str:
    if rec_df is None or rec_df.empty:
        return ""

    rec_df = rec_df.head(top_n)
    blocks = []
    for i, r in rec_df.iterrows():
        blocks.append(
            f"[{len(blocks)+1}] {r['대학명']} / {r['학과명']} ({r['지역']})\n"
            f"- 내신판단: {r['판단']} (사용자 vs 평균내신 {r['평균내신']})\n"
            f"- 지표: 학문평판={r['학문평판']}, 취업평판={r['취업평판']}, 교육밀도={r['교육밀도']}, "
            f"교수연구력={r['교수연구력']}, 국제화={r['국제화']}, 졸업성과={r['졸업성과']}, 순위={r['학교순위']}\n"
            f"- 검색점수: sim={r['유사도']:.3f}, bonus={r['보너스']:.3f}, score={r['최종점수']:.3f}"
        )
    return "\n\n".join(blocks)


# Extract grade level hints from the user's message for summary personalization.
def extract_grade_level(text: str) -> str:
    if re.search(r"고등학교\\s*1학년|고1", text):
        return "고1"
    if re.search(r"고등학교\\s*2학년|고2", text):
        return "고2"
    if re.search(r"고등학교\\s*3학년|고3", text):
        return "고3"
    if re.search(r"졸업", text):
        return "졸업"
    return "미상"


# Extract preferred region keywords from the user's message for summary personalization.
def extract_region(text: str) -> str:
    region_map = {
        "서울": "서울",
        "경기": "경기/인천",
        "인천": "경기/인천",
        "강원": "강원",
        "충청": "충청",
        "대전": "충청",
        "세종": "충청",
        "전라": "전라",
        "광주": "전라",
        "경상": "경상",
        "부산": "경상",
        "대구": "경상",
        "울산": "경상",
        "제주": "제주",
    }
    for keyword, region in region_map.items():
        if keyword in text:
            return region
    return "미상"


# Infer interest tags from user text to drive summary charts and skills.
def infer_interests(text: str) -> list[str]:
    keywords = [
        ("컴퓨터", "컴퓨터"),
        ("소프트웨어", "소프트웨어"),
        ("프로그래밍", "프로그래밍"),
        ("AI", "AI"),
        ("인공지능", "AI"),
        ("의학", "의학"),
        ("간호", "간호"),
        ("약학", "약학"),
        ("경영", "경영"),
        ("경제", "경제"),
        ("디자인", "디자인"),
        ("예술", "예술"),
    ]
    interests = []
    for key, label in keywords:
        if key.lower() in text.lower() or key in text:
            if label not in interests:
                interests.append(label)
    return interests


# Extract a short career goal phrase when explicitly stated by the user.
def infer_career_goal(text: str) -> str:
    match = re.search(r"(?:되고\\s*싶|꿈|목표|희망).*", text)
    return match.group(0) if match else ""


# Convert raw chat history into user/assistant pairs for summary display.
def build_conversation_pairs(history: list[dict]) -> list[dict]:
    pairs = []
    pending_user = None
    for message in history:
        if message.get("role") == "user":
            pending_user = message.get("content", "")
        elif message.get("role") == "assistant" and pending_user:
            pairs.append({"user": pending_user, "assistant": message.get("content", "")})
            pending_user = None
    return pairs


# Parse the JSON-only summary response, trimming any stray text.
def parse_summary_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


# Create lightweight trend series based on inferred interests.
def build_industry_trends(interests: list[str]) -> list[dict]:
    if not interests:
        return []
    base = 3600
    trends = []
    for idx, year in enumerate(["2020", "2021", "2022", "2023", "2024"]):
        trends.append({
            "year": year,
            "demand": min(95, 70 + idx * 6),
            "salary": base + idx * 150,
        })
    return trends


# Create lightweight skill requirements based on inferred interests.
def build_skill_requirements(interests: list[str]) -> list[dict]:
    if "AI" in interests or "컴퓨터" in interests or "소프트웨어" in interests or "프로그래밍" in interests:
        return [
            {"skill": "프로그래밍", "importance": 95},
            {"skill": "수학", "importance": 85},
            {"skill": "논리적 사고", "importance": 90},
            {"skill": "문제해결", "importance": 88},
            {"skill": "영어", "importance": 75},
        ]
    if "경영" in interests or "경제" in interests:
        return [
            {"skill": "데이터 분석", "importance": 88},
            {"skill": "커뮤니케이션", "importance": 85},
            {"skill": "문제해결", "importance": 82},
        ]
    if "디자인" in interests:
        return [
            {"skill": "창의성", "importance": 90},
            {"skill": "사용자 이해", "importance": 85},
            {"skill": "도구 활용", "importance": 80},
        ]
    return []


 


# ======================================================
# 6) API
# ======================================================
class ChatRequest(BaseModel):
    question: str
    session_id: str


VALID_INTENTS = {"RECOMMEND", "SCHOOL_INFO", "EXPLAIN_SYSTEM", "OTHER"}

def normalize_intent(raw: str) -> str:
    if not raw:
        return "OTHER"
    s = raw.strip().upper()
    s = s.split()[0].replace(":", "").replace("-", "_")
    return s if s in VALID_INTENTS else "OTHER"


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    session_id = req.session_id.strip()

    # 1) 히스토리
    history = get_history(session_id)
    history_text = history_to_text(history)

    # 2) 의도 분류 + 보정
    raw_intent = intent_chain.invoke({"question": question})
    intent = normalize_intent(raw_intent)

    # 3) 검색 (RECOMMEND / SCHOOL_INFO 공통)
    majors = search_major_contextual(question, top_k=20)
    use_rag = evaluate_retrieval(majors)

    # ===============================
    # 의도별 처리
    # ===============================
    if intent == "OTHER":
        answer = (
            "이 챗봇은 대학·학과 추천을 위한 상담 챗봇이에요.\n"
            "예를 들면:\n"
            "- 서울에 있는 컴퓨터공학과 추천해줘\n"
            "- 내신 3.2인데 수도권 공대 가능할까?\n"
            "처럼 질문해 주세요."
        )

    elif intent == "EXPLAIN_SYSTEM":
        answer = (
            "이 시스템은 질문을 분석해 의도를 분류하고,\n"
            "학과 임베딩 검색 + 내신 데이터 + 학교 평판 지표를 결합해\n"
            "현실적인 대학·학과 추천을 제공해요."
        )

    elif intent == "SCHOOL_INFO":
        # ✅ RAG가 부실하면: RAG 없이 GPT로 일반 안내
        if not use_rag:
            answer = fallback_chain.invoke({
                "history": history_text,
                "question": question,
            }).strip()

        else:
            # RAG가 충분할 때만: 기존처럼 top-1로 context 구성
            t = majors.iloc[0]
            context_text = (
                f"[1] {t.get('대학명','')} / {t.get('학과명','')} ({t.get('지역','')})\n"
                f"- 학과특성: {t.get('학과특성','정보 없음')}\n"
                f"- 계열: {t.get('표준분류계열(중)','정보 없음')}"
            )

            answer = answer_chain.invoke({
                "history": history_text,
                "question": question,
                "intent": intent,
                "context": context_text
            }).strip()

    else:  # RECOMMEND
        # ✅ RAG가 부실하면: RAG 없이 GPT로 일반 상담 + 추가 질문
        if not use_rag:
            answer = fallback_chain.invoke({
                "history": history_text,
                "question": question,
            }).strip()

        else:
            user_grade = extract_grade(question)

            if user_grade is not None and majors is not None and not majors.empty:
                rec_df = attach_score(majors, user_grade)
                context_text = build_context_text(rec_df, top_n=5)
            else:
                # 내신 없을 때: 최소 정보만
                blocks = []
                for _, r in majors.head(5).iterrows():
                    blocks.append(
                        f"- {r.get('대학명','')} / {r.get('학과명','')} ({r.get('지역','')})"
                    )
                context_text = "\n".join(blocks)

            answer = answer_chain.invoke({
                "history": history_text,
                "question": question,
                "intent": intent,
                "context": context_text
            }).strip()

    # 4) 히스토리 저장
    append_history(session_id, "user", question)
    append_history(session_id, "assistant", answer)

    return {
        "answer": answer,
        "intent": intent,
    }


# 프론트 요약 화면에 필요한 데이터를 생성하는 API다.
# 대화 이력과 질문을 바탕으로 추천 결과를 요약하고,
# LLM 실패 시에는 규칙 기반 대체 결과를 만든다.
@app.post("/recommendation/summary", response_model=RecommendationSummaryResponse)
def recommendation_summary(req: RecommendationSummaryRequest):
    question = req.question.strip()
    session_id = req.session_id.strip()

    history = get_history(session_id)
    history_text = history_to_text(history)

    majors = search_major_contextual(question, top_k=20)
    if majors is not None and not majors.empty:
        majors = rerank_with_filters(majors, question)

    user_grade = extract_grade(question)
    if user_grade is not None and majors is not None and not majors.empty:
        rec_df = attach_score(majors, user_grade)
    else:
        rec_df = pd.DataFrame()

    payload = build_recommendation_summary_payload(question, history, history_text, rec_df, user_grade)
    return RecommendationSummaryResponse(**payload)


# 요약 API의 핵심 페이로드 생성 함수다.
# LLM이 JSON을 반환하면 이를 파싱해 사용하고, 파싱 실패 시 규칙 기반 결과로 대체한다.
def build_recommendation_summary_payload(
    question: str,
    history: list[dict],
    history_text: str,
    rec_df: pd.DataFrame,
    user_grade: float | None,
) -> dict:
    # 추천 후보 중 상위 항목을 텍스트로 요약해 LLM 입력 컨텍스트로 제공한다.
    context_text = build_context_text(rec_df, top_n=5) if rec_df is not None else ""
    raw = summary_chain.invoke({
        "history": history_text,
        "question": question,
        "context": context_text,
    }).strip()
    try:
        return parse_summary_json(raw)
    except Exception:
        return build_summary_fallback(question, history, rec_df, user_grade)


# LLM이 JSON을 반환하지 못하거나 파싱 실패 시 사용되는 규칙 기반 요약 생성기다.
# 추천 후보 데이터와 질문 정보를 조합해 화면에서 사용할 수 있는 구조를 만든다.
def build_summary_fallback(
    question: str,
    history: list[dict],
    rec_df: pd.DataFrame,
    user_grade: float | None,
) -> dict:
    # 질문에서 관심 키워드를 추정해 요약 화면의 프로필/트렌드 데이터에 반영한다.
    interests = infer_interests(question)
    user_profile = {
        "interests": interests,
        "strengths": [],
        "careerGoal": infer_career_goal(question),
        "region": extract_region(question),
        "gradeLevel": extract_grade_level(question),
    }

    departments = []
    # 추천 후보가 있는 경우 상위 5개를 요약 항목으로 변환한다.
    if rec_df is not None and not rec_df.empty:
        for _, row in rec_df.head(5).iterrows():
            score_value = row.get("최종점수") or row.get("유사도") or 0.75
            base_score = float(score_value) * 100

            # 내신 정보가 있으면 평균 내신과의 격차를 점수에 반영해 적합도를 조정한다.
            avg_grade = row.get("평균내신")
            avg_grade_value = avg_grade if isinstance(avg_grade, (int, float)) else None
            if user_grade is not None and avg_grade_value is not None:
                grade_gap = user_grade - avg_grade_value
                gap_penalty = min(12, max(-6, grade_gap * 8))
                base_score -= gap_penalty

            match_score = int(min(95, max(65, base_score)))
            # 취업/학문 평판을 활용해 취업률과 평균 연봉을 추정치로 구성한다.
            employment_score = safe_float(row.get("취업평판"))
            if employment_score is None:
                employment_rate = max(60, min(92, match_score - 3))
            else:
                employment_rate = int(min(95, max(55, employment_score)))

            research_score = safe_float(row.get("학문평판"))
            base_salary = 3000 + (match_score - 65) * 30
            if research_score is not None:
                base_salary += int((research_score - 60) * 6)
            avg_salary = int(max(2800, min(4800, base_salary)))

            # 경쟁률은 적합도와 반비례하도록 단순화해 시각화용 값으로 만든다.
            competition_rate = round(4.0 + (95 - match_score) / 15, 1)
            required_grade = f"{avg_grade_value:.1f}등급" if avg_grade_value is not None else "정보 없음"
            major_name = row.get("학과명", "") or "추천 학과"
            university_name = row.get("대학명", "") or "추천 대학"
            region_name = row.get("지역", "")
            track = row.get("계열") or row.get("표준분류계열(중)") or ""
            major_feature = row.get("학과특성") or ""
            schedule = row.get("주야구분") or ""
            level = row.get("판단") or "적정"
            summary = (
                f"{university_name} {major_name}는 {region_name} 지역 기준으로 {level} 지원에 해당합니다. "
                "학과 특성과 성적 적합도를 함께 고려해 추천했습니다."
            )
            detail_bits = []
            if track:
                detail_bits.append(f"계열: {track}")
            if major_feature:
                detail_bits.append(f"학과 특성: {major_feature}")
            if schedule:
                detail_bits.append(f"운영: {schedule}")
            if detail_bits:
                summary = f"{summary} " + " / ".join(detail_bits) + "."
            departments.append({
                "name": major_name,
                "university": university_name,
                "matchScore": match_score,
                "employmentRate": employment_rate,
                "averageSalary": avg_salary,
                "competitionRate": competition_rate,
                "requiredGrade": required_grade,
                "description": summary,
                "relatedJobs": infer_related_jobs(major_name),
                "websiteUrl": row.get("홈페이지", ""),
            })

    # 대화 이력을 요약 화면에 표시할 수 있도록 사용자/어시스턴트 페어로 구성한다.
    pairs = build_conversation_pairs(history)
    initial_pairs = pairs[:3]
    after_pairs = pairs[3:6]

    return {
        "departments": departments,
        "userProfile": user_profile,
        "initialConversations": initial_pairs,
        "afterRecommendationConversations": after_pairs,
        "industryTrends": build_industry_trends(interests),
        "skillRequirements": build_skill_requirements(interests),
    }

