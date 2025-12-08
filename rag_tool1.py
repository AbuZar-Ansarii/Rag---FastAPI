from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
load_dotenv()



# -------------------------------
# 1. Load Documents
# -------------------------------
loader = TextLoader(r"E:\pandas\rag tools\MCP_Explanation.txt")  
docs = loader.load()
# -------------------------------
# 2. Split into Chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# -------------------------------
# 3. Create Embeddings
# -------------------------------
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# -------------------------------
# 4. Store in Vector Database (FAISS)
# -------------------------------
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

# -------------------------------
# 5. RAG PipelineS
# -------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# -------------------------------
# 6. FIXED: PromptTemplate (NOT ChatPromptTemplate)
# -------------------------------
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are a helpful assistant. Use ONLY the following context to answer.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}"
    ),
)

# -------------------------------
# 7. Correct chains
# -------------------------------
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# -------------------------------
# 8. Chat Loop
# -------------------------------
print("Chat with your RAG (type 'q' to exit):")
while True:
    user_q = input("\nYou: ")
    if user_q.lower() == "q":
        break

    result = rag_chain.invoke({"input": user_q})
    print("\nRAG:", result["answer"])

print("RAG tool loaded successfully.")