from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.agents import initialize_agent, AgentType

from mm_rag.agents.mm_embedder import Embedder
from mm_rag.config.config import config
from mm_rag.models import dynamodb, s3bucket, vectorstore
from mm_rag.pipelines.retrievers import Retriever
from mm_rag.processing.handlers import ImgHandler
from mm_rag.agents.tools import make_retrieve_tool
from mm_rag.agents.flows import run_chatbot
from mm_rag.agents.vlm_langchain import VLM

VectorStore = vectorstore.PineconeVectorStore(
    Embedder(),
    config['pinecone']['api_key'],
    config['pinecone']['index_name'],
    config['pinecone']['namespace'],
    config['pinecone']['hosting_cloud'],
    config['pinecone']['cloud_region']
  )

bucket = s3bucket.BucketService(s3bucket.create_bucket(
  config['aws']['bucketname']
))
dynamo = dynamodb.DynamoDB()
handler = ImgHandler()
embedder = Embedder()
retriever = Retriever(
      VectorStore, dynamo, bucket, embedder, handler
    )
retriever_tool = make_retrieve_tool(retriever)


if __name__ == "__main__":
  endpoint = HuggingFaceEndpoint(  # type: ignore
    repo_id = "Qwen/Qwen2.5-VL-32B-Instruct",
    verbose=True
  )

  chat = ChatHuggingFace(llm=endpoint, verbose=True)

  run_chatbot(retriever, chat, handler, bucket)