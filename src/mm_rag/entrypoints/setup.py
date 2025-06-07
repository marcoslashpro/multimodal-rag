from mm_rag.models import vectorstore, s3bucket, dynamodb
from mm_rag.config.config import config
from mm_rag.logging_service.log_config import create_logger
from mm_rag.agents.mm_embedder import Embedder
from mm_rag.processing.handlers import ImgHandler
from mm_rag.pipelines.retrievers import Retriever
from mm_rag.pipelines.uploaders import UploaderFactory
from mm_rag.processing.files import FileFactory
from mm_rag.processing.processors import ProcessorFactory
from mm_rag.pipelines.pipes import Piper
from mm_rag.agents.vlm import VLM
from huggingface_hub import InferenceClient
from mm_rag.utils import get_secret


logger = create_logger(__name__)


vector_store = vectorstore.PineconeVectorStore(
    Embedder(),
    get_secret()['pinecone_api_key'],
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
  vector_store, dynamo, bucket, embedder, handler
)

piper = Piper(
    uploader_factory=UploaderFactory(),
    processor_factory=ProcessorFactory(),
    retriever=retriever,
    file_factory=FileFactory(),
    owner='user123',
    embedder=embedder,
    dynamo=dynamo,
    vector_store=vector_store,
    s3=bucket,
    img_handler=handler
  )
client = InferenceClient(model="Qwen/Qwen2.5-VL-7B-Instruct")
vlm = VLM(model=client)