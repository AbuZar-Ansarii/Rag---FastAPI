import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import StrOutputParser
load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key,
)

response = llm.invoke(
    "what do you mean by central linmit theorem",)

print(response.content)