import io
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mm_rag.models.dynamodb import DynamoDB
  from mm_rag.models.vectorstore import PineconeVectorStore
  from mm_rag.models.s3bucket import BucketService
  from mm_rag.agents.mm_embedder import Embedder

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

from mm_rag.logging_service.log_config import create_logger
from mm_rag.exceptions import MalformedResponseError
import mm_rag.datastructures as ds

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


logger = create_logger(__name__)


class Retriever(BaseRetriever):
  def __init__(
      self,
      vector_store: 'PineconeVectorStore',
      dynamo: 'DynamoDB',
      bucket: 'BucketService',
      embedder: 'Embedder',
      top_k: int = 3
  ) -> None:
    super().__init__()
    self._vector_store = vector_store
    self._ddb = dynamo
    self._bucket = bucket
    self._top_k = top_k
    self._embedder = embedder

  def retrieve(self, query: str) -> list[Document]:
    logger.debug(f'Embedding query: {query}')
    retrieved_docs: list[Document] = []

    for modality in ds.Collection:
      embedded_query = self._embedder.embed(query, modality)

      collection = self._vector_store.namespace + modality.value

      retrieved = self._vector_store.index.query(
        top_k=self._top_k,
        vector=embedded_query,
        namespace=collection,
        include_metadata=True
      )
      logger.debug(f"retrieved: {retrieved}")

      if not retrieved:
        raise MalformedResponseError(
          f"Unable to gather response from VectorStore"
        )

      retrieved_docs.extend(self.transform_response_to_docs(retrieved))

    logger.debug(F"Retrieved docs: {retrieved_docs}")

    return retrieved_docs

  def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
    retrieved = self.retrieve(query)

    return retrieved

  def transform_response_to_docs(self, pinecone_response) -> list[Document]:
    docs: list[Document] = []
    content_buffer = io.BytesIO()

    for match in pinecone_response.get('matches'):
      match_id = match.get('id') or match.get('_id')
      if not match_id:
        raise MalformedResponseError(
          f"Malformed Pinecone Response, expected to find `id` field in matches, but none was found"
        )

      metadata = match.get('metadata')
      if not metadata:
        raise MalformedResponseError(
          f"Malformed PineconeResponse, expected to find `metadata` field in matches, but none was found"
        )

      page_content = metadata.get('text') or metadata.get('chunk_text')
      if not page_content:
        try:
          page_content = self._bucket.download_to_buffer(match_id, content_buffer).read().decode()
        except UnicodeDecodeError:
          page_content = self._bucket.generate_presigned_url(match_id)

          # raise MalformedResponseError(
          #   f"Malformed PineconeResponse, expected to find either `chunk_text` or `text` in response, but none was found."
          # )

      docs.append(
        Document(
          page_content=page_content,
          id=match_id,
          metadata=metadata
        )
      )

    return docs

