from typing import Annotated

from fastapi import APIRouter, UploadFile, Depends

from mm_rag.app.utils import write_file_to_lambda_path
from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints import upload_file, setup
from mm_rag.app.dependencies import auth_pat_dependency, HTTPAuthorizationCredentials
from mm_rag.app.utils import authorize


upload_router = APIRouter()
logger = create_logger(__name__)


@upload_router.post("/upload-file/")
def add_file(file: UploadFile, auth_pat: Annotated[HTTPAuthorizationCredentials, Depends(auth_pat_dependency)]):
  user = authorize(setup.dynamo, auth_pat.credentials)

  path = write_file_to_lambda_path(file)

  try:
    upload_file(path, user.user_id)

  except FileNotFoundError as e:
    return {
      "status": 404,
      "error": {
        "message": f"Generated file path: {path} from file: {file}, but we are not able to locate it on the cloud"
      }
    }
  except ValueError as e:
    return {
      "status": 400,
      "error": {
        "message": f"A validation error occured while processing the file {file.filename}: {e}"
      }
    }
  except RuntimeError as e:
    return {
      "status": 400,
      "error": {
        "message": f"A error occured while processing the file {file.filename}: {e}"
      }
    }

  return {
    "status": 200,
    "body": {
      "message": f"Upload of {file.filename} successfull!"
    }
  }
