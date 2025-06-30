from typing import Annotated

from fastapi import APIRouter, Depends, responses, HTTPException
from pydantic import BaseModel

from mm_rag.exceptions import MessageError, MissingResponseContentError
from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints import run_chatbot
from mm_rag.entrypoints.setup import vlm, bucket, dynamo, factory
from mm_rag.api.dependencies import auth_pat_dependency, HTTPAuthorizationCredentials
from mm_rag.api.utils import authorize
from mm_rag.api.models import Query


logger = create_logger(__name__)
chat_router = APIRouter()


@chat_router.post('/chat')
def chat(chat_input: Query, auth_pat: Annotated[HTTPAuthorizationCredentials, Depends(auth_pat_dependency)]):
  user = authorize(dynamo, auth_pat.credentials)

  try:
    response = run_chatbot(
      chat_input.query,
      factory.get_retriever(user.user_id),
      vlm,
      bucket
    )
  except MessageError as e:
    raise HTTPException(status_code=422, detail=str(e))

  except MissingResponseContentError as e:
    raise HTTPException(
      status_code=500,
      detail="The server was unable to generate a proper response for query: %s, full error: %s" %
             (chat_input.query, str(e))
    )


  return responses.JSONResponse(
    content=response
  )