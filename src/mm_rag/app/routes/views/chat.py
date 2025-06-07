from fastapi import APIRouter
from pydantic import BaseModel, Field

from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints import run_chatbot
from mm_rag.entrypoints.setup import retriever, vlm, handler, bucket

from uuid import uuid4

logger = create_logger(__name__)
chat_router = APIRouter()


class ChatInput(BaseModel):
  query: str
  id: str = uuid4().hex[:5]

@chat_router.post('/chat')
def chat(chat_input: ChatInput):
  try:
    response = run_chatbot(
      chat_input.query,
      retriever,
      vlm,
      handler,
      bucket
    )
  except Exception as e:
    msg = f"Error while processing queryId: {chat_input.id}: queryContent: {chat_input.query}"
    logger.error(msg + f': Error: {e}')
    return {
      "status": 500,
      "error": {
        "message": msg,
        "error": e
      }
    }
  
  logger.debug(f"Returning response: {response}")

  return {
    "status": 200,
    "message": {
      "body": response
    }
  }