import hashlib

from mm_rag.logging_service.log_config import create_logger
from mm_rag.exceptions import MissingItemError
from mm_rag.models.dynamodb import DynamoDB
from mm_rag.exceptions import FileNotValidError

from fastapi import HTTPException

from pydantic import BaseModel, ValidationError

import os


logger = create_logger(__name__)
LAMBDA_DIR = '/tmp/'


def write_file_to_lambda_path(file: bytes, filename: str) -> str:

  file_path = os.path.join(LAMBDA_DIR, filename)

  try:
    with open(file_path, 'wb') as f:
      f.write(file)

  except FileNotFoundError as e:
    logger.error(f"Error while converting the provided file {filename} to a serverless-friendly file: {e}")
    raise FileNotValidError(f"Error while converting the provided file {filename} to a serverless-friendly file: {e}")

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
      detail=f"Expected to find a user with both userId and PAT keys, instead found: {user}"
    )

  return auth_user
