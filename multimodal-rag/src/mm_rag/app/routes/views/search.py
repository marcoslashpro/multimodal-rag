from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from mm_rag.entrypoints import query_vectorstore, setup
from mm_rag.logging_service.log_config import create_logger
from mm_rag.app.dependencies import auth_pat_dependency, HTTPAuthorizationCredentials
from mm_rag.app.utils import authorize

from uuid import uuid4


logger = create_logger(__name__)
search_router = APIRouter()


class SearchQuery(BaseModel):
  query: str
  id: str = uuid4().hex[:5]


@search_router.post('/search')
def search(query: SearchQuery, auth_pat: Annotated[HTTPAuthorizationCredentials, Depends(auth_pat_dependency)]):
  logger.debug(f"Authorizing: {auth_pat}")
  user = authorize(setup.dynamo, auth_pat.credentials)

  try:
    retrieved = query_vectorstore(query.query, user.user_id)
  except Exception as e:
    msg = f"Error while querying the vectorstore with queryId: {query.id}: queryContent: {query.query}"
    logger.error(msg + str(e))
    return {
      "status": 500,
      "error": {
        "message": msg,
        "code": str(e)
      }
    }

  return {
    "status": 200,
    "message": {
      "body": [doc.to_json() for doc in retrieved]
    }
  }