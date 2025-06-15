from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from mm_rag.agents.chatbot_flow import State

from mm_rag.logging_service.log_config import create_logger
from mm_rag.agents.agent_utils import (
  MissingResponseContentError,
)


logger = create_logger(__name__)


def chatbot(state: 'State'):
  vlm = state['vlm']
  last_message = state['messages'][-1]

  try:
    response = vlm.invoke([last_message])

  except MissingResponseContentError as e:
    return {
      "messages": [
        {
          "role": "assistant",
          "content": str(e)
        }
      ]
    }

  return {
    "messages": [
      {
        "role": "assistant",
        "content": response.content
      }
    ]
  }
