from . import setup
import os
import asyncio

from mm_rag.logging_service.log_config import create_logger
from mm_rag.agents.chatbot_flow import run_chatbot

from langchain_core.documents import Document

from mm_rag.exceptions import ObjectDeletionError, MalformedResponseError

logger = create_logger(__name__)


async def upload_file(file_input: str, namespace: str) -> None:
  if not os.path.exists(file_input):

    logger.error(f"Provided file path: {file_input} does not exist")
    raise FileNotFoundError(
      f"Provided file path: {file_input} does not exist"
    )

  logger.debug(f"Piping file: {file_input} into namespace: {namespace}")

  await setup.piper.pipe(file_input, namespace)

  logger.debug(f"Upload of file {file_input} was a success.")


def query_vectorstore(query_input: str, namespace: str) -> list[Document]:

  logger.info(f'Instantiating the retriever')

  logger.info(f"Querying the VectorStore with input: {query_input}")
  retriever = setup.factory.get_retriever(namespace)
  try:
    retrieved = retriever.invoke(query_input)
  except MalformedResponseError as e:
    raise MalformedResponseError() from e

  return retrieved


async def cleanup(namespace: str) -> None:
  try:
    async with asyncio.TaskGroup() as tg:
      tg.create_task(setup.factory.get_vector_store(namespace).aclean())
      tg.create_task(setup.bucket.adelete_all())

  except* ObjectDeletionError as eg:
    logger.error(f"Caught the following exceptions in the exception group: {eg.exceptions}")
    pass
