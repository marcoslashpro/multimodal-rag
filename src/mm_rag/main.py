from mm_rag.models import vectorstore, s3bucket, dynamodb
from mm_rag.config.config import config
from mm_rag.logging_service.log_config import create_logger
from mm_rag.agents.mm_embedder import Embedder
from mm_rag.agents.tools import make_retrieve_tool
from mm_rag.processing.handlers import ImgHandler
from mm_rag.pipelines.retrievers import Retriever
from mm_rag.pipelines.uploaders import UploaderFactory
from mm_rag.processing.files import FileFactory
from mm_rag.processing.processors import ProcessorFactory
from mm_rag.pipelines.pipes import Piper
from mm_rag.agents.vlm_langchain import VLM
from langchain_core.tools.retriever import create_retriever_tool

import os
import getpass


# os.environ['HUGGINGFACEHUB_API_TOKEN'] = getpass.getpass(
# "Enter your HuggingFace API Key: "
# )


logger = create_logger(__name__)

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

piper = Piper(
    uploader_factory=UploaderFactory(),
    processor_factory=ProcessorFactory(),
    retriever=retriever,
    file_factory=FileFactory(),
    owner='user123',
    embedder=embedder,
    dynamo=dynamo,
    vector_store=VectorStore,
    s3=bucket,
    img_handler=handler
  )
retriever_tool = make_retrieve_tool(retriever)
vlm = VLM([retriever_tool])


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
  print(retriever_tool.invoke(query_input))

def chat() -> None:
  prompt = str(input('You: '))

  if not isinstance(prompt, str):
    raise ValueError(
      f"I do not know how you did it, but the provided prompt is not of type string, got {type(prompt)}"
    )
  
  vlm.chat(prompt)


def main() -> None:
  while True:
    choice: str = input("Search/Upload/Chat/Clean: ").lower()
    if choice == 'search':
      query()
    elif choice == 'upload':
      add_file()
    elif choice == 'chat':
      chat()
    elif choice =='clean':
      VectorStore.clean()
      bucket.delete_all()
      dynamo.clean()
    else:
      print('Softly exiting...')
      quit()

if __name__ == "__main__":
  main()
