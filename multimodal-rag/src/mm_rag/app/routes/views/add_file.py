from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from mm_rag.app.utils import write_file_to_lambda_path
from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints import upload_file, setup
from mm_rag.app.dependencies import auth_pat_dependency, HTTPAuthorizationCredentials
from mm_rag.app.utils import authorize


upload_router = APIRouter()
logger = create_logger(__name__)


@upload_router.post("/upload-file/")
async def add_file(
    auth_pat: Annotated[HTTPAuthorizationCredentials, Depends(auth_pat_dependency)],
    file: UploadFile = File(...),
):
  logger.info(f"Upload file route hit.")
  user = authorize(setup.dynamo, auth_pat.credentials)

  filename = file.filename
  bytes_file = file.file.read()

  if filename is None:
    logger.warning(f"Found UploadFile object with no file name: {file}")
    raise AttributeError(
      f"No filename found for the given file {file}, please provide it."
    )

  logger.debug(f"Witing {filename} to lambda path")
  path = write_file_to_lambda_path(bytes_file, filename)

  try:
    logger.debug(f"Uploading file: {path}, with userId = {user.user_id}")
    await upload_file(path, user.user_id)

  except FileNotFoundError as e:
    return {
      "status": 404,
      "error": {
        "message": f"Generated file path: {path} from file: {file}, but we are not able to locate it on the cloud"
      },
    }
  except ValueError as e:
    return {
      "status": 400,
      "error": {
        "message": f"A validation error occured while processing the file {file.filename}: {e}"
      },
    }
  except RuntimeError as e:
    return {
      "status": 400,
      "error": {
        "message": f"A error occured while processing the file {file.filename}: {e}"
      },
    }

  return {
    "status": 200,
    "body": {"message": f"Upload of {file.filename} successfull!"},
  }
