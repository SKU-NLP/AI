import uuid
import pandas as pd
import streamlit as st
import requests

# ======================================================
# 기본 설정 streamlit run app.py
# ======================================================
st.set_page_config(page_title="대학 · 학과 추천 챗봇", page_icon="🎓")
st.title("🎓 대학 · 학과 추천 챗봇")

API_URL = "http://localhost:8000/chat"
SUMMARY_URL = "http://localhost:8000/recommendation/summary"

st.markdown(
    """
    <style>
      html, body, [class*="css"]  {
        font-family: "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      }
      .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.6rem 0 0.2rem 0;
      }
      .section-subtitle {
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0.4rem 0 0.2rem 0;
      }
      .profile-metric [data-testid="stMetricValue"] {
        font-size: 0.9rem;
      }
      .profile-metric [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
      }
      .dept-card {
        border: 1px solid #111827;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0 0.8rem 0;
        background: #111827;
      }
      .dept-title {
        font-size: 1.0rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #ffffff;
      }
      .dept-desc {
        color: #e5e7eb;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
      }
      .table-caption {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================
# 세션 상태 초기화
# ======================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "summary" not in st.session_state:
    st.session_state.summary = None


# ======================================================
# 탭 구성
# ======================================================
chat_tab, summary_tab = st.tabs(["대화", "요약 결과"])

with chat_tab:
    st.subheader("대화")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("예: 서울에 있는 컴퓨터공학과 추천해줘"):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.write(prompt)

        try:
            res = requests.post(
                API_URL,
                json={"question": prompt, "session_id": st.session_state.session_id},
                timeout=60
            )
            res.raise_for_status()
            data = res.json()
            reply = data.get("answer", "응답을 받지 못했어요.")

        except requests.exceptions.RequestException as e:
            reply = f"❌ 서버 오류: {e}"
        except ValueError:
            reply = "❌ 서버 응답을 해석할 수 없습니다."

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        with st.chat_message("assistant"):
            st.write(reply)

with summary_tab:
    st.subheader("요약 결과")
    st.caption("대화를 몇 번 진행한 뒤 요약을 생성하면 더 자연스러운 결과가 나옵니다.")

    summary_question = st.text_input("요약 생성에 사용할 질문", value="내신 2.7인데 수도권 컴퓨터공학과 추천해줘")

    if st.button("요약 생성", type="primary"):
        try:
            res = requests.post(
                SUMMARY_URL,
                json={"question": summary_question, "session_id": st.session_state.session_id},
                timeout=90
            )
            res.raise_for_status()
            st.session_state.summary = res.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ 서버 오류: {e}")
        except ValueError:
            st.error("❌ 서버 응답을 해석할 수 없습니다.")

    if st.session_state.summary:
        summary = st.session_state.summary

        st.markdown("<div class='section-title'>추천 학과</div>", unsafe_allow_html=True)
        for dept in summary.get("departments", []):
            st.markdown(
                "<div class='dept-card'>"
                f"<div class='dept-title'>{dept.get('university')} {dept.get('name')}</div>"
                f"<div class='dept-desc'>{dept.get('description', '')}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(4)
            cols[0].metric("매칭도", f"{dept.get('matchScore', 0)}%")
            cols[1].metric("취업률", f"{dept.get('employmentRate', 0)}%")
            cols[2].metric("평균 연봉", f"{dept.get('averageSalary', 0)}만원")
            cols[3].metric("경쟁률", f"{dept.get('competitionRate', 0)}:1")

        st.markdown("<div class='section-title'>사용자 프로필</div>", unsafe_allow_html=True)
        profile = summary.get("userProfile", {})
        prof_cols = st.columns(3)
        with prof_cols[0]:
            st.markdown("<div class='profile-metric'>", unsafe_allow_html=True)
            st.metric("학년", profile.get("gradeLevel", "-"))
            st.markdown("</div>", unsafe_allow_html=True)
        with prof_cols[1]:
            st.markdown("<div class='profile-metric'>", unsafe_allow_html=True)
            st.metric("희망 지역", profile.get("region", "-"))
            st.markdown("</div>", unsafe_allow_html=True)
        with prof_cols[2]:
            st.markdown("<div class='profile-metric'>", unsafe_allow_html=True)
            st.metric("진로 목표", profile.get("careerGoal", "-"))
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>관심사</div>", unsafe_allow_html=True)
        st.write(", ".join(profile.get("interests", [])) or "-")

        strengths = profile.get("strengths", [])
        if strengths:
            st.markdown("<div class='section-subtitle'>강점 과목</div>", unsafe_allow_html=True)
            strengths_df = pd.DataFrame(
                [{"과목": s.get("subject"), "점수": s.get("score")} for s in strengths]
            )
            st.bar_chart(strengths_df.set_index("과목"))
            st.dataframe(strengths_df, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>산업 트렌드</div>", unsafe_allow_html=True)
        trends = summary.get("industryTrends", [])
        if trends:
            trends_df = pd.DataFrame(trends)
            chart_df = trends_df.set_index("year")
            st.line_chart(chart_df)
            st.dataframe(trends_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("<div class='table-caption'>표시할 트렌드 데이터가 없습니다.</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>역량 요구</div>", unsafe_allow_html=True)
        skills = summary.get("skillRequirements", [])
        if skills:
            skills_df = pd.DataFrame(skills)
            st.bar_chart(skills_df.set_index("skill"))
            st.dataframe(skills_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("<div class='table-caption'>표시할 역량 데이터가 없습니다.</div>", unsafe_allow_html=True)


st.divider()
st.subheader("질문 예시")
st.write(
    "- 내신 2.7인데 수도권 컴퓨터공학과 추천해줘\n"
    "- 고3이고 소프트웨어 쪽 관심 있어. 취업 잘 되는 학교 알려줘\n"
    "- 성적은 3등급 초반이고 경영학과 생각 중이야\n"
    "- 제주/강원 쪽으로 지역 제한해서 추천해줘"
)
