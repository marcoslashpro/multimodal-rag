from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
  from mm_rag.agents.mm_embedder import Embedder
  from mm_rag.models.dynamodb import DynamoDB
  from mm_rag.models.s3bucket import BucketService
  from mm_rag.models.vectorstore import VectorStoreFactory
  from mm_rag.pipelines.uploaders import UploaderFactory
  from mm_rag.processing.files import FileFactory
  from mm_rag.processing.handlers import ImgHandler
  from mm_rag.processing.processors import ProcessorFactory
  from mm_rag.pipelines.retrievers import RetrieverFactory

from langchain_core.documents import Document

from mm_rag.logging_service.log_config import create_logger


logger = create_logger(__name__)


class Piper:
  def __init__(
      self,
      uploader_factory: 'UploaderFactory',
      processor_factory: 'ProcessorFactory',
      retriever_factory: 'RetrieverFactory',
      file_factory: 'FileFactory',
      embedder: 'Embedder',
      dynamo: 'DynamoDB',
      vector_store_factory: 'VectorStoreFactory',
      s3: 'BucketService',
      img_handler: 'ImgHandler'
  ) -> None:
    self.img_handler = img_handler
    self.embedder = embedder
    self._uploader_factory = uploader_factory
    self._processor_factory = processor_factory
    self._retriever_facory = retriever_factory
    self._file_factory = file_factory
    self._ddb = dynamo
    self._vector_store_factory = vector_store_factory
    self._s3 = s3

  def _lazy_init(self, file_path: str, namespace: str) -> None:
    self.processor = self._processor_factory.get_processor(file_path, self.embedder, self.img_handler)
    self.file = self._file_factory.get_file(
        file_path,
        namespace,
        self.processor,
    )
    self._vector_store = self._vector_store_factory.get_vector_store(namespace)
    self.uploader = self._uploader_factory.get_uploader(
        file_path=file_path,
        dynamo=self._ddb,
        vector_store=self._vector_store,
        s3=self._s3,
        handler=self.img_handler
      )

    self.retriever = self._retriever_facory.get_retriever(self._vector_store)

    logger.debug(f"Inizitialized piper with:\nHandler: {self.img_handler}\nOwner: {namespace}\nEmbedder: {self.embedder}\n"
                 f"Bucket: {self._s3}\nDynamo: {self._ddb}, VectorStore: {self._vector_store}\nUploader: {self.uploader}"
                 f"\nProcessor: {self.processor}\nRetriever: {self.retriever}")


  def run_upload(self, file_path: str, namespace: str) -> None:
    self._lazy_init(file_path, namespace)

    docs = self.processor.process(self.file)
    logger.debug(f"Processed docs from file {file_path}: Docs: {docs}")

    logger.info(f"Upserting {self.file} and {docs} into VectorStore")
    self.uploader.upload_in_vector_store(self.file, docs)

    logger.info(f"Upserting {self.file} and {docs} into Bucket")
    self.uploader.upload_in_bucket(self.file, docs)

  def run_retrieval(self, query: str, namespace) -> list[Document]:
    return self._retriever_facory.get_retriever(self._vector_store_factory.get_vector_store(namespace)).invoke(query)