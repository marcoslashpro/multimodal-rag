from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from mm_rag.agents.chatbot_flow import State

from mm_rag.logging_service.log_config import create_logger


logger = create_logger(__name__)


def router(state: 'State'):
  is_retrieval_required = state.get('is_retrieval_required')
  if is_retrieval_required:
    logger.debug(f"Routing to 'retrieve'")
    return {"next": "retrieve"}

  logger.debug(f"Routing to 'chatbot'")
  return {"next": "chatbot"}
