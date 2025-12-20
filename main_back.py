from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
load_dotenv()
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI   
from langgraph.prebuilt import ToolNode, tools_condition

# special imports
from typing import Optional, Any
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools import DuckDuckGoSearchRun


CURRENT_RETRIEVER: Optional[Any] = None

search_tool = DuckDuckGoSearchRun()


def build_retriever_from_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()

# tools

@tool
def pdf_rag(query: str) -> str:
    """
    Answer questions using the uploaded PDF document.
    Use this tool only when the question depends on the document.
    """
    if CURRENT_RETRIEVER is None:
        return "No PDF uploaded yet."

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=(
            "Answer ONLY using the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}"
        ),
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(CURRENT_RETRIEVER, doc_chain)

    result = rag_chain.invoke({"input": query})
    return result["answer"]

@tool
def web_search(query: str) -> str:
    """this tool can search anything on web for current information."""
    results = search_tool.run(query)
    return results

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools([pdf_rag, web_search])



# ***************** nodes ******************


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chatbot(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

tools = [pdf_rag, web_search]
tool_node = ToolNode(tools)



graph = StateGraph(ChatState)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chatbot")

graph.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph.add_edge("tools", "chatbot")

app = graph.compile()

# ************* Streamlit app ********************

import streamlit as st
import tempfile
from langchain_core.messages import HumanMessage

st.title("Multi-Tool RAG Assistant")

uploaded_pdf = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    CURRENT_RETRIEVER = build_retriever_from_pdf(pdf_path)
    st.success("PDF indexed successfully")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask a question")

if user_input:
    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    result = app.invoke({
        "messages": st.session_state.messages
    })

    st.session_state.messages = result["messages"]

    st.chat_message("assistant").write(
        result["messages"][-1].content
    )

