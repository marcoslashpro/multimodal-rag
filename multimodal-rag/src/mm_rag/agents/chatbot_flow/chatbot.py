from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from mm_rag.agents.chatbot_flow import State

from mm_rag.logging_service.log_config import create_logger
from mm_rag.exceptions import (
  MissingResponseContentError,
)


logger = create_logger(__name__)


def chatbot(state: 'State'):
  vlm = state['vlm']
  last_message = state['messages'][-1]

  response = vlm.invoke([last_message])

  return {
    "messages": [
      {
        "role": "assistant",
        "content": response.content
      }
    ]
  }
