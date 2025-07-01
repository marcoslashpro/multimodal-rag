from typing import Any
import boto3
from botocore.exceptions import ClientError
from langchain_core.embeddings.embeddings import Embeddings

import json

from mm_rag.logging_service.log_config import create_logger


logger = create_logger(__name__)


class Embedder(Embeddings):
  def __init__(
      self,
      model_id: str = 'amazon.titan-embed-image-v1'
  ) -> None:
    self.client = boto3.client("bedrock-runtime", region_name='eu-central-1')
    self.model_id = model_id

  # Add `def embed_audio(self, audio_input: ?) -> list[float]`
  # Add `def embed_video(self, video_input: ?) -> list[float]`

  def embed_query(self, text: str) -> list[float]:
    request_body = json.dumps({
          "inputText": text
        })

    try:
      response: dict[str, Any] = self.client.invoke_model(
        modelId=self.model_id,
        body = request_body,
        accept="application/json",
        contentType="application/json"
      )
    except ClientError as e:
      raise

    return json.loads(response['body'].read())['embedding']

  def embed_img(self, base64_encoded_img: str) -> list[float]:
    request_body = json.dumps({
      "inputImage": base64_encoded_img
    })

    try:
      logger.debug(f'Invoking the model for the response')
      response = self.client.invoke_model(
        modelId=self.model_id,
        body = request_body
      )
    except ClientError as e:
      logger.error(f'Call to the model failed: {e}')
      raise

    logger.debug(f"Got response: {response}")
    return json.loads(response['body'].read())['embedding']

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    embedded_docs: list[list[float]] = []

    if isinstance(texts, list):
      for text in texts:
        embedded_docs.append(self.embed_query(text))

      return embedded_docs

    else:
      raise ValueError(
        f"Input must be a list[str], got {type(texts)}"
      )