from typing import Annotated

from fastapi import APIRouter, Depends

from mm_rag.entrypoints import cleanup, setup
from mm_rag.logging_service.log_config import create_logger
from mm_rag.app.dependencies import auth_pat_dependency, HTTPAuthorizationCredentials
from mm_rag.app.utils import authorize


logger = create_logger(__name__)
cleanup_router = APIRouter()


@cleanup_router.post('/cleanUp')
def clean(auth_pat: Annotated[HTTPAuthorizationCredentials, Depends(auth_pat_dependency)]):
  user = authorize(setup.dynamo, auth_pat.credentials)

  try:
    cleanup(user.user_id)
  except Exception as e:
    return {
      "satus": 500,
      "error": {
        "message": str(e)
      }
    }

  return {
    "satus": 200,
    "message": {
      "body": "Databases cleaned successfully!"
    }
  }