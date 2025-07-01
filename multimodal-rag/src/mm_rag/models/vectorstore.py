import asyncio
from typing import Literal

from mm_rag.logging_service.log_config import create_logger
from mm_rag.config.config import config
from mm_rag.agents.mm_embedder import Embedder
from mm_rag.exceptions import ObjectUpsertionError
import mm_rag.datastructures as ds

import pinecone
from pinecone import Pinecone, Vector
from pinecone import ServerlessSpec

from langchain_pinecone import PineconeVectorStore as lcPineconeVectorStore

from botocore.exceptions import ClientError

from mm_rag.exceptions import ObjectDeletionError, FileNotValidError, DocGenerationError
import pinecone.exceptions

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
    self.cloud = cloud
    self.region = region
    self.namespace = namespace

  @property
  def index(self):
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

  def add(self, file: ds.File) -> None:
    if not len(file.docs) == len(file.embeddings):
      raise ObjectUpsertionError(
        storage=ds.Storages.VECTORSTORE,
        msg=f"Length of docs and embeddings of file {file.metadata.file_id} do not match: "
          "{len(file.docs)} != {len(file.embeddings)}"
      )

    try:
      for doc, embeddings in zip(file.docs, file.embeddings):
        if not doc.id:
          raise ObjectUpsertionError(
            storage=ds.Storages.VECTORSTORE,
            msg=f"Invalid document generated, missing id for doc: {doc}"
          )

        self.index.upsert(
          [
            Vector(
              id=doc.id,
              values=embeddings,
              metadata=doc.metadata
          )],
          namespace=self._generate_full_namespace(file.metadata.collection)
        )
    except pinecone.exceptions.PineconeException as e:
      raise ObjectUpsertionError(
        storage=ds.Storages.VECTORSTORE,
        msg=str(e)
      ) from e

  def _generate_full_namespace(self, collection: str) -> str:
    return self.namespace + f"/{collection}"

  def add_image(
      self,
      encoded_img: str,
      metadata: ds.Metadata,
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
      raise ObjectUpsertionError(ds.Storages.VECTORSTORE) from e
    return True

  def clean(self) -> None:
    try:
      self.vector_store.delete(delete_all=True)
    except pinecone.PineconeException as e:
      raise ObjectDeletionError(ds.Storages.VECTORSTORE) from e

  async def aclean(self):
    await asyncio.to_thread(self.clean)

  def remove_object(self, id: str) -> None:
    self.vector_store.delete([id], namespace=self.namespace)


if __name__ == '__main__':
  pass