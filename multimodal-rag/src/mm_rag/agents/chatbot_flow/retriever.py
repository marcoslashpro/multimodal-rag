from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from mm_rag.agents.chatbot_flow import State

from mm_rag.logging_service.log_config import create_logger


logger = create_logger(__name__)


def retrieve(state: 'State'):
  retriever = state['retriever']
  last_message = state['messages'][-1]
  query = state['query']
  logger.debug(f"Running retrieval on input: {last_message}")

  retrieved: list = retriever.retrieve(last_message.content)
  logger.debug(f"Found in the VectorStore: {retrieved}")

  return {"retrieved": retrieved, 'query': query}
