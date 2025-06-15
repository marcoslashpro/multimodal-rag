from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from mm_rag.app.utils import write_file_to_lambda_path
from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints import upload_file, setup
from mm_rag.app.dependencies import auth_pat_dependency, HTTPAuthorizationCredentials
from mm_rag.app.utils import authorize
from mm_rag.exceptions import FileNotValidError, DocGenerationError, ImageTooBigError, ObjectUpsertionError


upload_router = APIRouter()
logger = create_logger(__name__)


@upload_router.post("/upload-file/")
async def add_file(
    auth_pat: Annotated[HTTPAuthorizationCredentials, Depends(auth_pat_dependency)],
    file: UploadFile = File(...),
):
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

  except (FileNotValidError, FileNotFoundError) as e:
    return {
      "status": 404,
      "error": {
        "message": f"Invalid file {file.filename}. Full error: {e}"
      },
    }
  except (ValueError, ImageTooBigError, DocGenerationError, RuntimeError) as e:
    return {
      "status": 400,
      "error": {
        "message": f"A validation error occured while processing the file {file.filename}: {e}"
      },
    }
  except (ObjectUpsertionError, ExceptionGroup) as e:
    return {
      "status": 500,
      "error": {
        "message": f"Error while uploading into {e.storage}. Upload in the other storage canceled."
      }
    }

  return {
    "status": 200,
    "body": {"message": f"Upload of {file.filename} successfull!"},
  }
