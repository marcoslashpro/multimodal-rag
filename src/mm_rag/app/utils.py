from mm_rag.app import lambda_dir
from mm_rag.logging_service.log_config import create_logger

from fastapi import UploadFile

import shutil
import os


logger = create_logger(__name__)


def write_file_to_lambda_path(file: UploadFile) -> str:
  if not file.filename:
    logger.debug(f'Found UploadFile object with no file name: {file}')
    raise AttributeError(
      f"No filename found for the given file {file}, please provide it."
    )

  file_path = os.path.join(lambda_dir, file.filename)

  try:
    with open(file_path, 'wb') as f:
      shutil.copyfileobj(file.file, f)
  except shutil.Error as e:
    raise RuntimeError(
      f"Error while converting the provided file {file.filename} to a serverless-friendly file: {e}"
    )

  return file_path