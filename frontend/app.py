import streamlit as st
import requests

import os
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.title("YouTube RAG Assistant")

if "token" not in st.session_state:
    st.session_state.token = None

if st.session_state.token is None:

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login to your account")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            response = requests.post(
                f"{API_URL}/login",
                json={
                    "username": username,
                    "password": password
                }
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.token = data["access_token"]
                st.session_state.user_id = data["user_id"]

                history_response = requests.get(
                    f"{API_URL}/history",
                    headers={
                        "Authorization": f"Bearer {st.session_state.token}"
                    }
                )

                st.session_state.messages = []
                if history_response.status_code == 200:
                    rows = history_response.json()
                    for row in rows:
                        st.session_state.messages.append(
                            {
                                "role": row["role"],
                                "content": row["content"]
                            }
                        )
                st.success("Logged in")
                st.rerun()
            else:
                st.error("Login failed")

    with tab2:
        st.subheader("Create a new account")
        reg_username = st.text_input("New Username", key="reg_user")
        reg_password = st.text_input("New Password", type="password", key="reg_pass")

        if st.button("Register"):
            response = requests.post(
                f"{API_URL}/register",
                json={
                    "username": reg_username,
                    "password": reg_password
                }
            )
            if response.status_code == 200:
                st.success("Account created successfully! Please switch to the Login tab to log in.")
            else:
                st.error(f"Registration failed: {response.text}")

    st.stop()
st.divider()

youtube_url = st.text_input(
    "Youtube URL"
)

if st.button("Ingest Video"):

    response = requests.post(
        f"{API_URL}/ingest",
        headers={
            "Authorization":
            f"Bearer {st.session_state.token}"
        },
        json={
            "url": youtube_url
        }
    )

    if response.status_code == 200:

        st.session_state.messages = []

        st.success("Video ingested")

    else:
        st.error(response.text)


if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if (msg["role"] == "assistant" and "citations" in msg):

            with st.expander("Sources"):

                for citation in msg["citations"]:

                    st.markdown(
                        f"[{citation['timestamp']}]({citation['youtube_link']})"
                    )

prompt = st.chat_input(
    "Ask a question"
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    response = requests.post(
        f"{API_URL}/chat",
        headers={
            "Authorization":
            f"Bearer {st.session_state.token}"
        },
        json={
            "question": prompt
        }
    )

    if response.status_code == 200:
        data = response.json()

        answer = data["answer"]
        citations = data["citations"]

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations
            }
        )
    else:
        st.error(f"Error from backend: {response.text}")

    st.rerun()


