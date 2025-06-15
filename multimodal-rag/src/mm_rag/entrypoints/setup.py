from mm_rag.models import vectorstore, s3bucket, dynamodb
from mm_rag.config.config import config
from mm_rag.logging_service.log_config import create_logger
from mm_rag.agents.mm_embedder import Embedder
from mm_rag.pipelines.retrievers import RetrieverFactory
from mm_rag.pipelines.pipes import pipe
from mm_rag.agents.vlm import VLM
from huggingface_hub import InferenceClient
from mm_rag.utils import get_secret


logger = create_logger(__name__)


vector_store_factory = vectorstore.VectorStoreFactory(
    Embedder(),
    get_secret()['pinecone_api_key'],
    config['pinecone']['index_name'],
    config['pinecone']['hosting_cloud'],
    config['pinecone']['cloud_region']
  )

bucket = s3bucket.BucketService(s3bucket.create_bucket(
  config['aws']['bucketname']
))
dynamo = dynamodb.DynamoDB()
embedder = Embedder()
retriever_factory = RetrieverFactory(
  dynamo, bucket, embedder
)

client = InferenceClient(model="Qwen/Qwen2.5-VL-7B-Instruct", api_key=get_secret()['hf_token'])
vlm = VLM(model=client)