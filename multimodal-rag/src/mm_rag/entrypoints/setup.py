from mm_rag.models import vectorstore, s3bucket, dynamodb
from mm_rag.config.config import config
from mm_rag.logging_service.log_config import create_logger
from mm_rag.agents.mm_embedder import Embedder
from mm_rag.pipelines.pipes import Piper, ComponentFactory
from mm_rag.agents.vlm import VLM
from huggingface_hub import InferenceClient
from mm_rag.utils import get_secret


logger = create_logger(__name__)


bucket = s3bucket.BucketService(s3bucket.create_bucket(
  config['aws']['bucketname']
))
dynamo = dynamodb.DynamoDB()
embedder = Embedder()

client = InferenceClient(model="Qwen/Qwen2.5-VL-7B-Instruct", api_key=get_secret()['hf_token'])
vlm = VLM(model=client)

factory = ComponentFactory(
    embedder=embedder,
    api_key=get_secret()['pinecone_api_key'],
    index_name=config['pinecone']['index_name'],
    cloud=config['pinecone']['hosting_cloud'],
    region=config['pinecone']['cloud_region'],
    dynamodb=dynamo,
    bucket=bucket
)

piper = Piper(
  factory=factory
)
