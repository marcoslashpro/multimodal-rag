from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mm_rag.entrypoints import query_vectorstore, setup
from mm_rag.logging_service.log_config import create_logger
from mm_rag.api.dependencies import auth_pat_dependency, HTTPAuthorizationCredentials
from mm_rag.api.utils import authorize
from mm_rag.api.models import Query
from mm_rag.exceptions import MalformedResponseError

from uuid import uuid4


logger = create_logger(__name__)
search_router = APIRouter()


@search_router.post('/search')
def search(query: Query, auth_pat: Annotated[HTTPAuthorizationCredentials, Depends(auth_pat_dependency)]):
  logger.debug(f"Authorizing: {auth_pat}")
  user = authorize(setup.dynamo, auth_pat.credentials)

  if len(query.query) == 0:
    raise HTTPException(
      status_code=422,
      detail=f"Query must not be empty"
    )

  try:
    retrieved = query_vectorstore(query.query, user.user_id)
  except MalformedResponseError as e:
    msg = f"Error while querying the vectorstore with query: {query.query}"
    logger.error(msg + str(e))

    raise HTTPException(
      status_code=500,
      detail=msg + str(e)
    )

  return JSONResponse(
    status_code=200,
    content=[doc.model_dump(mode='json') for doc in retrieved]
  )
