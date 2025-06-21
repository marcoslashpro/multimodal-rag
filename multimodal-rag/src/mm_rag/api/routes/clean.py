from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from mm_rag.entrypoints import cleanup, setup
from mm_rag.logging_service.log_config import create_logger
from mm_rag.api.dependencies import auth_pat_dependency, HTTPAuthorizationCredentials
from mm_rag.api.utils import authorize
from mm_rag.exceptions import ObjectDeletionError


logger = create_logger(__name__)
cleanup_router = APIRouter()


@cleanup_router.post('/cleanUp')
async def clean(auth_pat: Annotated[HTTPAuthorizationCredentials, Depends(auth_pat_dependency)]):
  user = authorize(setup.dynamo, auth_pat.credentials)

  await cleanup(user.user_id)

  return Response(
    content="Databases cleaned successfully!"
  )
