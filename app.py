from fastapi import FastAPI, Request
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
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
from pydantic import BaseModel

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)

# Text loader
loader = TextLoader("./whatspie.md")
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create the embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load into Chroma
db = Chroma.from_documents(docs, embedding_function)

message = """
Kamu adalah Customer Support AI dari whatspie yang siap menjawab pertanyaan apapun, jika pertanyaan nya kurang jelas, tolong tanyakan lagi dan jangan menjawab sesuatu yang kurang pasti.\
kemudian jawaban yang kamu berikan harus dalam bahasa indonesia \
Kamu hanya bisa menjawab tentang context berikut 

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", message),
    MessagesPlaceholder(variable_name="chat_histories"),
    ("human", "{input}")
])

retriever = db.as_retriever()

stuff_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
)

chain = create_retrieval_chain(
    retriever,
    stuff_chain
)

chat_histories = {}

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_histories": chat_history
    })
    return response


app = FastAPI()

class FromUser(BaseModel):
    jid: str
    name: str

class WhatspieBody(BaseModel):
    message_id: str
    from_user: FromUser
    message: str

# Dictionary to store chat histories
chat_histories = {}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/hooks")
def handle_hooks(item: WhatspieBody):
    user_id = item.from_user.jid
    # Retrieve or create chat history for the user
    if user_id not in chat_histories:
        chat_histories[user_id] = [
            HumanMessage(content="Hai, nama saya " + item.from_user.name)
        ]
    chat_history = chat_histories[user_id]
    response = process_chat(chain, item.message, chat_history)

    # Update chat history
    chat_history.append(HumanMessage(content=item.message))
    chat_history.append(response['answer'])

    return {
        "type": "chat",
        "body": response['answer'],
        "simulation": "true",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)