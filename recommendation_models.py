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
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
