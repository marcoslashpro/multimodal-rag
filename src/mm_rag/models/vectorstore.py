from typing import TYPE_CHECKING

from mm_rag.logging_service.log_config import create_logger
from mm_rag.config.config import config
from mm_rag.agents.mm_embedder import Embedder
if TYPE_CHECKING:
  from mm_rag.processing.base import Metadata

import pinecone
from pinecone import Pinecone, Vector
from pinecone import ServerlessSpec

from langchain_pinecone import PineconeVectorStore as lcPineconeVectorStore

from botocore.exceptions import ClientError


logger = create_logger(__name__)


class PineconeVectorStore:
  def __init__(
    self,
    embedder: Embedder,
    api_key: str,
    index_name: str,
    namespace: str,
    cloud: str,
    region: str
	) -> None:
    self.embedder = embedder
    self.api_key = api_key
    self.index_name = index_name
    self.namespace = namespace
    self.cloud = cloud
    self.region = region

  @property
  def index(self) -> pinecone.data.index.Index:
    try:
      pc = Pinecone(api_key=self.api_key)

      if not pc.has_index(self.index_name):
        pc.create_index(
          name=self.index_name,
          dimension=1024,
          metric='cosine',
          spec=ServerlessSpec(
            cloud=self.cloud,
            region=self.region
          )
        )
      return pc.Index(config['pinecone']['index_name'])
    except ClientError as e:
      raise
    except Exception as e:
      raise

  @property
  def vector_store(self) -> lcPineconeVectorStore:
      try:
       return lcPineconeVectorStore(
          index=self.index,
          embedding=self.embedder,
          namespace=self.namespace,
          index_name=self.index_name
        )
      # TODO Improve error handling
      except Exception as e:
        raise

  def add_image(
      self,
      encoded_img: str,
      metadata: "Metadata",
      file_id: str,
    ) -> bool:
    try:
      values: list[float] = self.embedder.embed_img(encoded_img)
      # logger.debug(f"Values for encded img: {values}")
      logger.debug(f"Dimension of input: {len(values)}")
      if not len(values) == 1024:  # Index dimensions
        raise ValueError(
          f"Expected 1024 dimensions for input, but got {len(values)}"
        )

      # Langchain expects a 'text' metadata key.
      # We cannot unfortunately encode the entire img and set it in the 'text' key.
      # We therefore store the fileId in order to then retrieve the image from the s3 Bucket.
      metadata.__dict__['text'] = file_id

      self.index.upsert(
        [
          Vector(
          id=file_id,
          values=values,
          metadata=metadata.__dict__
          )
        ],
        namespace=self.namespace
      )
    except pinecone.PineconeException as e:
      logger.error(e)
      return False
    return True

  def clean(self) -> None:
    self.vector_store.delete(delete_all=True)

if __name__ == '__main__':
  pass