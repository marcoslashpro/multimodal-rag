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

import os

logger = create_logger(__name__)

VectorStore = vectorstore.PineconeVectorStore(
    Embedder(),
    config['pinecone']['api_key'],
    config['pinecone']['index_name'],
    config['pinecone']['namespace'],
    config['pinecone']['hosting_cloud'],
    config['pinecone']['cloud_region']
  )

bucket = s3bucket.BucketService(s3bucket.Bucket())
dynamo = dynamodb.DynamoDB()
handler = ImgHandler()

piper = Piper(
    uploader_factory=UploaderFactory(),
    processor_factory=ProcessorFactory(),
    retriever=Retriever(
      VectorStore, dynamo, bucket, handler
    ),
    file_factory=FileFactory(),
    owner='user123',
    embedder=Embedder(),
    dynamo=dynamo,
    vector_store=VectorStore,
    s3=bucket,
    img_handler=handler
  )

def add_file() -> None:
  file_input: str = input("Insert file path: ")

  if not os.path.exists(file_input):

    logger.error(f"Provided file path: {file_input} does not exist")
    raise FileNotFoundError()

  piper.run_upload(file_input)

def query() -> None:

  logger.info(f'Instatiating the retriever')

  query_input: str = input('Search: ')

  logger.info(f"Querying the VectorStore with input: {query_input}")
  piper.run_retrieval(query_input)


def main() -> None:
  while True:
    choice: str = input("Search/Upload/Clean: ").lower()
    if choice == 'search':
      query()
    elif choice == 'upload':
      add_file()
    elif choice =='clean':
      VectorStore.clean()
      bucket.delete_all()
      dynamo.clean()
    else:
      print('Softly exiting...')
      quit()

if __name__ == "__main__":
  main()