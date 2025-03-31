from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["PINECONE_API_KEY"] = "pcsk_3Jcjfc_Pu6ddn3gygeGxuYq7ZU3zqSJ9s2WoULe6EKvcZDJKpShU9aGnyarSzMncvL9s52"
os.environ["OPENAI_API_KEY"] = "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"

extracted_data = load_pdf("Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key="pcsk_3Jcjfc_Pu6ddn3gygeGxuYq7ZU3zqSJ9s2WoULe6EKvcZDJKpShU9aGnyarSzMncvL9s52")

index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)