from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, responses

from mm_rag.api.utils import write_file_to_lambda_path,authorize
from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints import upload_file, setup
from mm_rag.api.dependencies import auth_pat_dependency, HTTPAuthorizationCredentials
from mm_rag.exceptions import FileNotValidError, DocGenerationError, ImageTooBigError, ObjectUpsertionError, StorageError


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

  if len(bytes_file) == 0:
    raise HTTPException(
      status_code=404,
      detail="Cannot upload an empty file: {filename}"
    )

  if filename is None:
    logger.warning(f"Found UploadFile object with no file name: {file}")
    raise HTTPException(
      status_code=404,
      detail="No filename found for the given file {file}, please provide it."
    )

  try:
    logger.debug(f"Writing {filename} to lambda path")
    path = write_file_to_lambda_path(bytes_file, filename)

    logger.debug(f"Uploading file: {path}, with userId = {user.user_id}")
    await upload_file(path, user.user_id)

  except (FileNotValidError, FileNotFoundError) as e:
    raise HTTPException(
      status_code=404,
      detail=f"Invalid file {file.filename}. Full error: {e}"
    )

  except (ValueError, DocGenerationError, RuntimeError) as e:
    raise HTTPException(
      status_code=400,
      detail=f"A validation error occurred while processing the file {file.filename}: {e}"
    )
  except ImageTooBigError as e:
    raise HTTPException(
      status_code=413, detail=str(e)
    )

  except ObjectUpsertionError as e:
    raise HTTPException(
      status_code=500,
      detail=f"Error while uploading into {e.storage}. Upload in the other storage canceled."
    )
  except StorageError as e:
    raise HTTPException(
      status_code=500,
      detail=str(e)
    )

  return responses.JSONResponse(
    status_code=200,
    content={"message": f"Upload of {file.filename} successful!"}
  )
