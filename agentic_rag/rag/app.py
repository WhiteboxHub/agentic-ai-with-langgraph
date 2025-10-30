import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.retriever import MilvusRetriever
from llm_selector import get_llm
from utils.prompts import prompt

# ----------------------------
# 🔧 Initialize Components
# ----------------------------
@st.cache_resource
def init_pipeline():
    llm = get_llm(provider="groq")
    retriever = MilvusRetriever(
        collection_name="documents_chunks",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        milvus_host="localhost",
        milvus_port="19530"
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

rag_chain = init_pipeline()

# ----------------------------
# 🎨 Streamlit UI
# ----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 AI RAG Chatbot")
st.caption("Powered by Milvus + LangChain + Groq/OpenAI")

# Chat input
user_input = st.chat_input("Ask something about your data...")

# Maintain conversation context
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response.get("answer", "I couldn’t find anything relevant.")
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
