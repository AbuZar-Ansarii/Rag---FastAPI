import streamlit as st
import requests

UPLOAD_URL = 'http://127.0.0.1:8000/upload'
QUERY_URL = 'http://127.0.0.1:8000/query'

st.set_page_config(page_title="RAG Chat Bot", page_icon="💬")
st.title("💬 RAG Chat Bot")

# --- Upload Section ---
with st.expander("📄 Upload a PDF Document", expanded=True):
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Uploading and processing..."):
            files = {"file": (uploaded_file.name, uploaded_file, 'application/pdf')}
            response = requests.post(UPLOAD_URL, files=files)
            if response.status_code == 200:
                st.success("File uploaded and processed successfully!")
            else:
                st.error(f"Failed to upload file: {response.text}")

# --- Chat Section ---
st.markdown("---")
st.header("🗨️ Chat with your documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def display_chat():
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f"**🧑‍💻 You:** {entry['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {entry['content']}")
            # Show sources if available
            if entry.get("sources"):
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(entry["sources"], 1):
                        st.markdown(f"**Document {i}:**")
                        st.write(doc.get("page_content", ""))
                        st.write("Metadata:", doc.get("metadata", {}))

display_chat()

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question:", key="user_input")
    submitted = st.form_submit_button("Send")
    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = requests.post(QUERY_URL, json={"query": user_input})
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer found.")
                sources = data.get("source_documents", [])
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Failed to get answer: {response.text}"
                })
        st.rerun()
