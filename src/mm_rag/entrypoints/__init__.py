from . import setup
import os

from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints.setup import retriever
from mm_rag.agents.flows import run_chatbot

from langchain_core.documents import Document


logger = create_logger(__name__)


def upload_file(file_input: str) -> None:
  if not os.path.exists(file_input):

    logger.error(f"Provided file path: {file_input} does not exist")
    raise FileNotFoundError()

  setup.piper.run_upload(file_input)


def query_vectorstore(query_input: str) -> list[Document]:

  logger.info(f'Instatiating the retriever')

  logger.info(f"Querying the VectorStore with input: {query_input}")
  retrieved = retriever.invoke(query_input)

  return retrieved
