from fastapi import APIRouter
from pydantic import BaseModel, Field

from mm_rag.entrypoints import query_vectorstore
from mm_rag.logging_service.log_config import create_logger

from uuid import uuid4


logger = create_logger(__name__)
search_router = APIRouter()


class SearchQuery(BaseModel):
  query: str
  id: str = uuid4().hex[:5]


@search_router.post('/search')
def search(query: SearchQuery):
  try:
    retrieved = query_vectorstore(query.query)
  except Exception as e:
    msg = f"Error while querying the vectorstore with queryId: {query.id}: queryContent: {query.query}"
    logger.error(msg)
    return {
      "status": 500,
      "error": {
        "message": msg
      }
    }

  return {
    "status": 200,
    "message": {
      "body": retrieved
    }
  }