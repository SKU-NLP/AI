import streamlit as st
import requests

# ======================================================
# 기본 설정
# ======================================================
st.set_page_config(page_title="대학 · 학과 추천 챗봇", page_icon="🎓")
st.title("🎓 대학 · 학과 추천 챗봇")

API_URL = "http://localhost:8000/chat"


# ======================================================
# 세션 상태 초기화
# ======================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# ======================================================
# 이전 대화 출력
# ======================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ======================================================
# 입력창
# ======================================================
if prompt := st.chat_input("예: 서울에 있는 컴퓨터공학과 추천해줘"):
    # 사용자 메시지 저장 & 출력
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.write(prompt)

    # ==================================================
    # API 호출
    # ==================================================
    try:
        res = requests.post(
            API_URL,
            json={"question": prompt},   # ✅ FastAPI와 맞춤
            timeout=60
        )

        res.raise_for_status()
        data = res.json()
        reply = data.get("answer", "응답을 받지 못했어요.")

    except requests.exceptions.RequestException as e:
        reply = f"❌ 서버 오류: {e}"
    except ValueError:
        reply = "❌ 서버 응답을 해석할 수 없습니다."

    # ==================================================
    # 챗봇 응답 저장 & 출력
    # ==================================================
    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
    with st.chat_message("assistant"):
        st.write(reply)
