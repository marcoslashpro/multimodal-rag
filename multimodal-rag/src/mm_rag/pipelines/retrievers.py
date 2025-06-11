import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from mm_rag.models.dynamodb import DynamoDB
  from mm_rag.models.vectorstore import PineconeVectorStore, VectorStoreFactory
  from mm_rag.models.s3bucket import BucketService
  from mm_rag.agents.mm_embedder import Embedder

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

from mm_rag.processing.handlers import ImgHandler
from mm_rag.logging_service.log_config import create_logger

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


logger = create_logger(__name__)


class RetrieverFactory:
  def __init__(
    self,
    dynamo: 'DynamoDB',
    bucket: 'BucketService',
    embedder: 'Embedder',
    handler: 'ImgHandler | None' = None,
    top_k: int = 3,
  ) -> None:
    self.dynamo = dynamo
    self.bucket = bucket
    self.embedder = embedder
    self.handler = handler
    self.top_k = top_k

  def get_retriever(self, vector_store: 'PineconeVectorStore') -> 'Retriever':
    return Retriever(
      vector_store,
      self.dynamo,
      self.bucket,
      self.embedder,
      self.handler,
      self.top_k
    )


class Retriever(BaseRetriever):
  def __init__(
      self,
      vector_store: 'PineconeVectorStore',
      dynamo: 'DynamoDB',
      bucket: 'BucketService',
      embedder: 'Embedder',
      handler: 'ImgHandler | None' = None,
      top_k: int = 3
  ) -> None:
    super().__init__()
    self._vector_store = vector_store
    self._ddb = dynamo
    self._bucket = bucket
    self._top_k = top_k
    self._embedder = embedder
    if handler:
      self._handler = handler

  def retrieve_and_display(self, query: str) -> None:
    retrieved_docs = self.retrieve(query)

    for doc in retrieved_docs:
      logger.debug('Match for query: %s' % doc)
      match_id = doc.id
      if not match_id:
        raise ValueError(
          f"Found doc with no id field: {doc}"
        )

      metadata = doc.metadata
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
        if not self._handler:
          raise ValueError(
            f"Retrieved an Image match, in order to display it, plase pass a Handler to the class."
          )

        logger.debug(f"Calling the handler to display the Img: {match_id}")
        self._handler.display(match_id, self._bucket)

      else:
        print(f"\n\n=====Chunk{match_id}=====\n")
        print(doc.metadata.get('text') or doc.metadata.get('chunk_text'))

  def retrieve(self, query: str) -> list[Document]:
    logger.debug(f'Embedding query: {query}')
    embedded_query: list[float] = self._embedder.embed_query(query)

    retrieved = self._vector_store.index.query(
      top_k=self._top_k,
      vector=embedded_query,
      namespace=self._vector_store.namespace,
      include_metadata=True
    )

    if not retrieved:
      raise RuntimeError(
        f"Unable to gather response from VectorStore"
      )

    retrieved_docs: list[Document] = self.transform_response_to_docs(retrieved)  #type: ignore[call-args]

    return retrieved_docs

  def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
    retrieved = self.retrieve(query)

    return retrieved

  def forward(self, query: str) -> str:  # type: ignore[override]
    retrieved_docs = self.retrieve(query)

    return '\n'.join([doc.page_content for doc in retrieved_docs])

  def transform_response_to_docs(self, pinecone_response) -> list[Document]:
    docs: list[Document] = []

    for match in pinecone_response['matches']:
      match_id = match.get('id') or match.get('_id')
      if not match_id:
        raise ValueError(
          f"Malformed Pinecone Response, expected to find `id` field in matches, but none was found"
        )

      metadata = match.get('metadata')
      if not metadata:
        raise ValueError(
          f"Malformed PineconeResponse, expected to find `metadata` field in matches, but none was found"
        )

      page_content = metadata.get('text') or metadata.get('chunk_text')
      if not page_content:
        raise ValueError(
          f"Malformed PineconeResponse, expected to find either `chunk_text` or `text` in response, but none was found."
        )

      docs.append(
        Document(
          page_content=page_content,
          id=match_id,
          metadata=metadata
        )
      )

    return docs

  @staticmethod
  def from_docs_to_string(docs: list[Document]) -> str:
    string_docs = ''
    for doc in docs:
      string_docs += json.dumps(doc.to_json()['kwargs']) + '\n\n' # type: ignore[call-args]

    return string_docs
