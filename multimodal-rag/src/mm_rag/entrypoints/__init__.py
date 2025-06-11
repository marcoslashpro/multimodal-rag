from . import setup
import os

from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints.setup import piper
from mm_rag.agents.chatbot_flow import run_chatbot

from langchain_core.documents import Document


logger = create_logger(__name__)


def upload_file(file_input: str, namespace: str) -> None:
  if not os.path.exists(file_input):

    logger.error(f"Provided file path: {file_input} does not exist")
    raise FileNotFoundError()

  setup.piper.run_upload(file_input, namespace)


def query_vectorstore(query_input: str, namespace: str) -> list[Document]:

  logger.info(f'Instatiating the retriever')

  logger.info(f"Querying the VectorStore with input: {query_input}")
  retrieved = piper.run_retrieval(query_input, namespace)

  return retrieved


def cleanup(namespace: str) -> None:
  setup.vector_store_factory.get_vector_store(namespace).clean()
  setup.bucket.delete_all()
  setup.dynamo.clean()
