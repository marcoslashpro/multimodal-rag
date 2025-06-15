from . import setup
import os

from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints.setup import pipe
from mm_rag.agents.chatbot_flow import run_chatbot

from langchain_core.documents import Document


logger = create_logger(__name__)


async def upload_file(file_input: str, namespace: str) -> None:
  if not os.path.exists(file_input):

    logger.error(f"Provided file path: {file_input} does not exist")
    raise FileNotFoundError(
      f"Provided file path: {file_input} does not exist"
    )

  await setup.pipe(
    file_input,
    namespace,
    setup.dynamo,
    setup.vector_store_factory,
    setup.bucket
  )


def query_vectorstore(query_input: str, namespace: str) -> list[Document]:

  logger.info(f'Instatiating the retriever')

  logger.info(f"Querying the VectorStore with input: {query_input}")
  vectorstore = setup.vector_store_factory.get_vector_store(namespace)
  retriever = setup.retriever_factory.get_retriever(vectorstore)
  retrieved = retriever.invoke(query_input)

  return retrieved


def cleanup(namespace: str) -> None:
  setup.vector_store_factory.get_vector_store(namespace).clean()
  setup.bucket.delete_all()
  setup.dynamo.clean()
