from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from mm_rag.models.dynamodb import DynamoDB
  from mm_rag.models.vectorstore import PineconeVectorStore
  from mm_rag.models.s3bucket import BucketService
  from pinecone.data import QueryResponse

from mm_rag.processing.handlers import ImgHandler
from mm_rag.logging_service.log_config import create_logger
from mm_rag.models import dynamodb, s3bucket, vectorstore

from PIL import Image, UnidentifiedImageError


logger = create_logger(__name__)


class Retriever:
  def __init__(
      self,
      vector_store: 'PineconeVectorStore',
      dynamo: 'DynamoDB',
      bucket: 'BucketService',
      handler: 'ImgHandler | None' = None,
      top_k: int = 5
  ) -> None:
    self.vector_store = vector_store
    self.ddb = dynamo
    self.bucket = bucket
    self.top_k = top_k
    if handler:
      self.handler = handler

  def retrieve_and_display(self, embedded_query: list[float]) -> None:
    response = self.retrieve(embedded_query)

    for match in response.get('matches'):
      logger.debug('Match for query: %s' % match)
      match_id = match.get('id')

      metadata = match.get('metadata')
      logger.debug(f"Found metadata for {match_id}")
      if not metadata:
        raise ValueError(
          f"No metadata key found from the given matches in the response."
        )

      file_type = metadata.get('fileType')
      logger.debug(F'File type for {match_id}: {file_type}')
      if not file_type:
        raise ValueError(
          f"No Type key in the metadata of the response"
        )

      if file_type in ['.jpeg', '.png', '.jpg', '.pdf']:
        if not self.handler:
          raise ValueError(
            f"Retrieved an Image match, in order to display it, plase pass a Handler to the class."
          )

        logger.debug(f"Calling the handler to display the Img: {match_id}")
        self.handler.display(match_id, self.bucket)

      else:
        print(f"\n\n=====Chunk{match_id}=====")
        print(match['metadata']['text'])

  def retrieve(self, embedded_query: list[float]) -> 'QueryResponse':
    response = self.vector_store.index.query(
      top_k=self.top_k,
      vector=embedded_query,
      namespace=self.vector_store.namespace,
      include_metadata=True
    )

    if not response:
      raise RuntimeError(
        f"Unable to gather response from VectorStore"
      )
    return response  # type: ignore