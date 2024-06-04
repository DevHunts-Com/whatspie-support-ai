from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from pydantic import BaseModel

load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

# Text loader
# load the document and split it into chunks
loader = TextLoader("./whatspie.md")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

message = """
Kamu adalah Customer Support AI dari whatspie yang siap menjawab pertanyaan apapun berdasarkan context berikut

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

retriever = db.as_retriever()

# chain =   prompt | model | retrieval_chain
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

app = FastAPI()

class FromUser(BaseModel):
    jid: str
    name: str

class WhatspieBody(BaseModel):
    message_id: str
    from_user: FromUser
    message: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/hooks")
def read_root(item: WhatspieBody):
    response = rag_chain.invoke(item.message)

    return {
        "type": "chat",
        "body": response.content,
        "simulation": "true",
    }