from flask import Flask, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
import os
import re

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

_missing = [
    name
    for name, value in (
        ("PINECONE_API_KEY", PINECONE_API_KEY),
        ("GROQ_API_KEY", GROQ_API_KEY),
    )
    if not value
]
if _missing:
    raise RuntimeError(
        "Missing required environment variable(s): "
        + ", ".join(_missing)
        + ". Copy .env.example to .env and fill in your keys."
    )


embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k":3})

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="deepseek-r1-distill-qwen-32b"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = (request.values.get("msg") or "").strip()
    if not msg:
        return "Please type a question.", 400

    app.logger.info("question: %s", msg)
    try:
        response = rag_chain.invoke({"input": msg})
    except Exception:
        app.logger.exception("the retrieval chain failed")
        return "Sorry, I could not answer that right now. Please try again.", 502

    answer = response.get("answer", "")
    cleaned_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    return cleaned_answer or "I don't have an answer for that."



if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=8080, debug=debug)
