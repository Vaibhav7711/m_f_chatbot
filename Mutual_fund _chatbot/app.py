# ==========================================
# MUTUAL FUND CHATBOT WEB APP
# ==========================================

import streamlit as st
from rag_mf import build_vector_store, ask_question

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Mutual Fund AI Assistant",
    page_icon="💬",
    layout="centered"
)

# ==========================================
# HEADER
# ==========================================

st.title("💬 Mutual Fund AI Assistant")

st.caption(
    "Ask factual questions about PPFAS Mutual Fund schemes (Factsheet: Jan 2026)"
)

# ==========================================
# LOAD VECTOR DATABASE
# ==========================================

@st.cache_resource
def load_database():
    return build_vector_store()

collection = load_database()

# ==========================================
# SCHEME SELECTOR
# ==========================================

scheme_map = {
    "Parag Parikh Liquid Fund": "PPLF",
    "Parag Parikh Flexi Cap Fund": "PPFCF",
    "Parag Parikh ELSS Tax Saver Fund": "PPTSF"
}

scheme_name = st.selectbox(
    "Select Scheme",
    list(scheme_map.keys())
)

scheme_code = scheme_map[scheme_name]

# ==========================================
# CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# ==========================================
# USER INPUT
# ==========================================

user_input = st.chat_input(
    "Ask a question about the selected scheme..."
)

if user_input:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Searching factsheet..."):

            response = ask_question(
                collection,
                user_input,
                scheme_code
            )

            st.write(response)

    # Save bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Disclaimer: This chatbot provides factual information from mutual fund factsheets. "
    "It does not provide investment advice."
)