import hashlib
import secrets

from mm_rag.app import lambda_dir
from mm_rag.logging_service.log_config import create_logger
from mm_rag.exceptions.models_exceptions import MissingItemError
from mm_rag.models.dynamodb import DynamoDB

from fastapi import UploadFile, HTTPException, Response

from pydantic import BaseModel, ValidationError

from botocore.exceptions import ClientError

import shutil
import os


logger = create_logger(__name__)


def write_file_to_lambda_path(file: bytes, filename: str) -> str:

  file_path = os.path.join(lambda_dir, filename)

  try:
    with open(file_path, 'wb') as f:
      f.write(file)

  except Exception as e:
    logger.error(f"Error while converting the provided file {filename} to a serverless-friendly file: {e}")
    raise HTTPException(500, f"Error while converting the provided file {filename} to a serverless-friendly file: {e}")

  return file_path


class AuthUser(BaseModel):
  user_id: str
  pat: str


def authorize(ddb: DynamoDB, token: str) -> AuthUser:
  encoded_token = hashlib.sha256(token.encode()).hexdigest()

  try:
    user = ddb.query_with_gsi(
      table_name='users',
      index_name="PAT-gsi-index",
      gsi_key="PAT-gsi",
      gsi_value=encoded_token
    )
  except MissingItemError:
    raise HTTPException(status_code=403, detail=f"No user found with token {token}, access denied")
  
  user = user[0]

  logger.debug(f"Authenticated user: {user}")

  user_id = user.get("userId")
  pat = user.get("PAT")

  try:
    auth_user = AuthUser(
      user_id=user_id,
      pat=pat
    )
  except ValidationError as e:
    raise HTTPException(
      status_code=500,
      detail=f"Exepected to find a user with both userId and PAT keys, instead found: {user}"
    )

  return auth_user
