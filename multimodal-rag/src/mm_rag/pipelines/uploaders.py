from typing import TYPE_CHECKING, Union

from mm_rag.models.dynamodb import DynamoDB
from mm_rag.models.s3bucket import BucketService
from mm_rag.models.vectorstore import PineconeVectorStore
from mm_rag.processing.handlers import ImgHandler
if TYPE_CHECKING:
  from mm_rag.models import dynamodb, vectorstore, s3bucket
  from mm_rag.processing.handlers import ImgHandler

import os
import asyncio

from abc import ABC, abstractmethod

from botocore.exceptions import ClientError

from langchain_core.documents import Document

import pinecone

from mm_rag.processing.files import TxtFile, ImgFile, PdfFile
from mm_rag.logging_service.log_config import create_logger
from mm_rag.exceptions.models_exceptions import ObjectUpsertionError

logger = create_logger(__name__)


class UploaderFactory:
  def get_uploader(
      self,
      file_path: str,
      dynamo: 'dynamodb.DynamoDB',
      vector_store: 'vectorstore.PineconeVectorStore',
      s3: 's3bucket.BucketService',
      handler: 'ImgHandler | None' = None
      ) -> Union['ImgUploader', 'PdfUploader', 'TxtUploader']:
    uploader = self.create_uploader(
      file_path,
      dynamo,
      vector_store,
      s3,
      handler
      )
    return uploader

  def create_uploader(
      self,
      file_path: str,
      dynamo: 'dynamodb.DynamoDB',
      vector_store: 'vectorstore.PineconeVectorStore',
      s3: 's3bucket.BucketService',
      handler: 'ImgHandler | None'
      ) -> Union['ImgUploader', 'PdfUploader', 'TxtUploader']:
    _, file_ext = os.path.splitext(file_path)

    if file_ext == '.txt':
      return TxtUploader(
        dynamodb=dynamo,
        vector_store=vector_store,
        bucket=s3,
        handler=handler
      )

    if file_ext in ['.pdf', '.docx']:
      return PdfUploader(
        dynamodb=dynamo,
        vector_store=vector_store,
        bucket=s3,
        handler=handler
      )

    if file_ext in ['.jpeg', '.jpg', '.png']:
      return ImgUploader(
        dynamodb=dynamo,
        vector_store=vector_store,
        bucket=s3,
        handler=handler
      )

    else:
      raise ValueError(
        f'File type {file_ext} not yet supported'
      )


class Uploader(ABC):
  def __init__(
      self,
      dynamodb: 'dynamodb.DynamoDB',
      vector_store: 'vectorstore.PineconeVectorStore',
      bucket: 's3bucket.BucketService',
      handler: 'ImgHandler | None' = None
  ) -> None:
    self.ddb = dynamodb
    self.vector_store = vector_store
    self.bucket = bucket
    if handler:
      self.handler = handler

  @abstractmethod
  def upload_in_vector_store(self, file: Union['TxtFile', 'ImgFile', 'PdfFile'], docs: list[Document]) -> bool:
    """
    This function must be implemented in order to upsert any object in the VectorStore
    """

  @abstractmethod
  def upload_in_bucket(self, file: Union['TxtFile', 'ImgFile', 'PdfFile'], docs: list[Document] | None = None) -> None:
    """
    Let the subclass define how to be upserted into the s3bucket
    """

  @abstractmethod
  def upload_in_dynamo(self, file: Union['TxtFile', 'ImgFile', 'PdfFile']) -> None:
    pass

  async def aupload_in_vector_store(self, file: Union['TxtFile', 'ImgFile', 'PdfFile'], docs: list[Document]):
    try:
      return await asyncio.to_thread(self.upload_in_vector_store, file, docs)

    except (AttributeError, ValueError, ObjectUpsertionError) as e:
      raise ObjectUpsertionError("PineconeVectorStore") from e

  async def aupload_in_bucket(self, file: Union['TxtFile', 'ImgFile', 'PdfFile'], docs: list[Document] | None = None):
    try:
      return await asyncio.to_thread(self.upload_in_bucket, file, docs)

    except (AttributeError, ValueError, ObjectUpsertionError) as e:
      raise ObjectUpsertionError("BucketService") from e


class TxtUploader(Uploader):
  def upload_in_vector_store(self, file: Union['TxtFile', 'ImgFile', 'PdfFile'], docs: list[Document]) -> bool:
    if not isinstance(file, TxtFile):
      raise ValueError(
        f'Expected file of type TxtFile, got {type(file)}'
      )

    try:
      logger.debug(f"upserting docs of {file.file_path} to the VectorStore")
      self.vector_store.vector_store.add_documents(docs)

      logger.debug("Done")

    except pinecone.PineconeException as e:
      logger.error(e)
      raise ObjectUpsertionError('PineconeVectorStore') from e 

    return True

  def upload_in_bucket(self, file: Union['TxtFile', 'ImgFile', 'PdfFile'], docs: list[Document] | None = None) -> None:
    logger.debug(f"Inserting {file.metadata.fileName} in {self.bucket.name}")

    if not isinstance(file, TxtFile):
      raise ValueError(
        f'Expected object of type TxtFile got {type(file)} for TxtPiper'
      )

    try:
      self.bucket.upload_object_from_path(
        file.file_path,
        file.file_id
        )
    except ClientError as e:
      logger.error(f"Error while upserting {file.metadata.fileName} in bucket {self.bucket.name}: {e}")
      raise ObjectUpsertionError('BucketService') from e 

    logger.debug("Done")

  def upload_in_dynamo(self, file: Union['TxtFile', 'ImgFile', 'PdfFile']) -> None:
    logger.debug(f"Inserting {file.metadata.fileName} in DynamoDb")

    if not isinstance(file, TxtFile):
      raise ValueError(
        f"Expected file of type TxtFile, got {type(file)}"
      )

    self.ddb.store_file(
      "files",
      file.file_id,
      file.owner,
      file.metadata.__dict__
    )

    logger.debug("Done")


class ImgUploader(Uploader):
  def __init__(self, dynamodb: DynamoDB, vector_store: PineconeVectorStore, bucket: BucketService, handler: ImgHandler | None) -> None:
    super().__init__(dynamodb, vector_store, bucket, handler)

    if not self.handler:
      raise AttributeError(
        f"Handler required for PdfUploader"
      )

  def upload_in_vector_store(self, file: Union['ImgFile', 'PdfFile', 'TxtFile'], docs: list[Document]) -> bool:
    if not isinstance(file, ImgFile):
      raise ValueError(
        f"expected file of type ImgFile, got {type(file)}"
      )

    try:
      logger.debug(f'Upserting img: {file.file_path} to the VectorStore')
      self.vector_store.add_image(file.encodings, file.metadata, file.file_id)
      logger.debug(f"Done.")

    except pinecone.PineconeException as e:
      logger.error("Error wile upserting the image %s to the VectorStore: %s" % file.file_path, e)
      raise ObjectUpsertionError('PineconeVectorStore') from e 

    return True

  def upload_in_bucket(self, file: Union['ImgFile', 'TxtFile', 'PdfFile'], docs: list[Document] | None = None) -> None:
    logger.debug(f"Inserting {file.metadata.fileName} in {self.bucket.name}")

    if not isinstance(file, ImgFile):
      raise ValueError(
        f'Expected file of type ImgFile, got {type(file)}'
      )

    img_buffer = self.handler.save_img_to_buffer(file.file_content)

    try:
      self.bucket.upload_object_from_file(
        img_buffer,
        file.file_id
        )

    except ClientError as e:
      logger.error(f"Error while upserting {file.metadata.fileName} in bucket {self.bucket.name}: {e}")
      raise ObjectUpsertionError('BucketService') from e 

    logger.debug("Done")

  def upload_in_dynamo(self, file: 'TxtFile | ImgFile | PdfFile') -> None:
    logger.debug(f"Inserting {file.metadata.fileName} in DynamoDb")

    if not isinstance(file, ImgFile):
      raise ValueError(
        f"Expected file of type ImgFile, got {type(file)}"
      )

    self.ddb.store_file(
      "files",
      file.file_id,
      file.owner,
      file.metadata.__dict__
    )

    logger.debug("Done")


class PdfUploader(Uploader):
  def __init__(self, dynamodb: DynamoDB, vector_store: PineconeVectorStore, bucket: BucketService, handler: ImgHandler | None) -> None:
    super().__init__(dynamodb, vector_store, bucket, handler)

    if not self.handler:
      raise AttributeError(
        f"Handler required for PdfUploader"
      )

  def upload_in_vector_store(self, file: Union['ImgFile', 'TxtFile', 'PdfFile'], docs: list[Document]) -> bool:
    if not isinstance(file, PdfFile):
      raise ValueError(
        f"Expected file of type PdfFile, got {type(file)}"
      )

    for i in range(len(file.file_content)):
      page_id = docs[i].id

      if not page_id:
        raise ValueError(
          f"In order to upload the given PdfFile to the VectorStore, "
          f"please provide valid ids in `docs`"
        )

      try:
        logger.debug(f"Upserting {file.file_path} to the VectorStore")
        self.vector_store.add_image(file.encodings[i], file.metadata, page_id)
        logger.debug(f"Done.")

      except pinecone.PineconeException as e:
        logger.error("Error wile upserting the pdf page %d %s to the VectorStore: %s" % i, file.file_path, e)
        raise ObjectUpsertionError('PineconeVectorStore') from e 

    logger.debug(f"All pages upserted.")
    return True

  def upload_in_bucket(self, file: Union['TxtFile', 'PdfFile', 'ImgFile'], docs: list[Document] | None = None) -> None:
    logger.debug(f"Inserting {file.metadata.fileName} in {self.bucket.name}")

    if not isinstance(file, PdfFile):
      raise ValueError(
        f"Expected file of type PdfFile, got {type(file)}"
      )
    if not docs:
      raise AttributeError(
        f"In order to upload PdfFile to the Bucket, please pass over the docs"
      )

    for i, page in enumerate(file.file_content):
      page_id = docs[i].id

      if not page_id:
        raise ValueError(
          f"In order to upload the given PdfFile to the VectorStore, "
          f"please provide valid ids in `docs`. Found doc {i+1} without id:\n{docs[i]}"
        )

      page_buffer = self.handler.save_img_to_buffer(page)

      try:
        self.bucket.upload_object_from_file(
          page_buffer,
          page_id
        )
      except ClientError as e:
        logger.error(f"Error while upserting {file.metadata.fileName} in bucket {self.bucket.name}: {e}")
        raise ObjectUpsertionError('BucketService') from e 

      logger.debug("Done")

  def upload_in_dynamo(self, file: 'TxtFile | ImgFile | PdfFile') -> None:
    logger.debug(f"Inserting {file.metadata.fileName} in DynamoDb")

    if not isinstance(file, PdfFile):
      raise ValueError(
        f"Expected file of type PdfFile, got {type(file)}"
      )

    self.ddb.store_file(
      "files",
      file.file_id,
      file.owner,
      file.metadata.__dict__
    )

    logger.debug("Done")
