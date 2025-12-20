"""
구조화된 추천 요청/응답 모델
"""
from pydantic import BaseModel
from typing import Optional, List


class StudentProfile(BaseModel):
    """학생 기본 정보"""
    grade_level: str  # "high1", "high2", "high3", "graduate"
    region: str  # "seoul", "gyeonggi", etc.
    grade_score: str  # "top", "high", "mid", "low"
    economic_status: Optional[str] = None  # "necessary", "prefer", "unnecessary"
    activities: Optional[str] = None  # 특기 및 수상 경력


class StudentInterests(BaseModel):
    """학생 흥미/적성 정보"""
    enjoyable_activities: str  # 질문 1: 즐거운 활동
    strengths: str  # 질문 2: 강점
    future_field: str  # 질문 3: 미래 희망 분야
    favorite_subjects: str  # 질문 4: 좋아하는 과목
    hobbies: str  # 질문 5: 여가 활동


class DetailedRecommendationRequest(BaseModel):
    """상세 추천 요청"""
    profile: StudentProfile
    interests: StudentInterests


class MajorRecommendation(BaseModel):
    """개별 학과 추천"""
    university: str
    major: str
    location: str
    match_score: float
    reason: str


class DetailedRecommendationResponse(BaseModel):
    """상세 추천 응답"""
    recommendations: List[MajorRecommendation]
    summary: str
    total_matches: int


# 기존 단순 모델 (호환성 유지)
# /chat API에서 사용하는 기본 요청 모델입니다.
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


# 요약 API에서 사용하는 요청 모델입니다. 세션 기반으로 대화 이력을 불러옵니다.
class RecommendationSummaryRequest(BaseModel):
    question: str
    session_id: str


# /chat API의 응답 스키마입니다.
class ChatResponse(BaseModel):
    answer: str
    matched_count: Optional[int] = None
    intent: Optional[str] = None


# 레이더 차트에 쓰는 과목별 강점 항목 DTO입니다.
class ProfileStrength(BaseModel):
    subject: str
    score: int


# 사용자 프로필 및 강점 요약 데이터 DTO입니다.
class UserProfileSummary(BaseModel):
    interests: List[str]
    strengths: List[ProfileStrength]
    careerGoal: str
    region: str
    gradeLevel: str


# 추천 학과 카드에 필요한 요약 정보 DTO입니다.
class DepartmentSummary(BaseModel):
    name: str
    university: str
    matchScore: int
    employmentRate: int
    averageSalary: int
    competitionRate: float
    requiredGrade: str
    description: str
    relatedJobs: List[str]
    websiteUrl: str


# 대화 시나리오의 발화 쌍 DTO입니다.
class ConversationPair(BaseModel):
    user: str
    assistant: str


# 산업 트렌드 차트에 필요한 연도별 지표 DTO입니다.
class IndustryTrendPoint(BaseModel):
    year: str
    demand: int
    salary: int


# 직무/전공 요구 역량 차트용 DTO입니다.
class SkillRequirement(BaseModel):
    skill: str
    importance: int


# 추천 결과 요약 데이터를 묶어 반환하는 응답 DTO입니다.
class RecommendationSummaryResponse(BaseModel):
    departments: List[DepartmentSummary]
    userProfile: UserProfileSummary
    initialConversations: List[ConversationPair]
    afterRecommendationConversations: List[ConversationPair]
    industryTrends: List[IndustryTrendPoint]
    skillRequirements: List[SkillRequirement]
