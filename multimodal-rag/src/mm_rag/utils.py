from typing import Literal
import boto3
from botocore.exceptions import ClientError

import json

def get_secret():
  secret_name = "prod/mm-rag/api_keys"
  region_name = "eu-central-1"

  # Create a Secrets Manager client
  client = boto3.client(
    service_name='secretsmanager',
    region_name=region_name
  )

  try:
    get_secret_value_response = client.get_secret_value(
      SecretId=secret_name
    )
  except ClientError as e:
    # For a list of exceptions thrown, see
    # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    raise e

  secret = get_secret_value_response['SecretString']

  return json.loads(secret)
