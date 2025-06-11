from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints import run_chatbot
from mm_rag.entrypoints.setup import vlm, handler, bucket, dynamo, retriever_factory, vector_store_factory
from mm_rag.app.dependencies import auth_pat_dependency, HTTPAuthorizationCredentials
from mm_rag.app.utils import authorize

from uuid import uuid4

logger = create_logger(__name__)
chat_router = APIRouter()


class ChatInput(BaseModel):
  query: str


@chat_router.post('/chat')
def chat(chat_input: ChatInput, auth_pat: Annotated[HTTPAuthorizationCredentials, Depends(auth_pat_dependency)]):
  user = authorize(dynamo, auth_pat.credentials)

  response = run_chatbot(
    chat_input.query,
    retriever_factory.get_retriever(vector_store_factory.get_vector_store(user.user_id)),
    vlm,
    handler,
    bucket
  )

  return {
    "status": 200,
    "message": {
      "body": response
    }
  }