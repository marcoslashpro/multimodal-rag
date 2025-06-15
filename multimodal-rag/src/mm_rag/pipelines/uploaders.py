from typing import TYPE_CHECKING

from mm_rag.models.dynamodb import DynamoDB
from mm_rag.models.s3bucket import BucketService
from mm_rag.models.vectorstore import PineconeVectorStore
if TYPE_CHECKING:
  from mm_rag.models import dynamodb, vectorstore, s3bucket

import asyncio

from abc import ABC, abstractmethod

from botocore.exceptions import ClientError

import pinecone

import mm_rag.pipelines.datastructures as ds
import mm_rag.pipelines.utils as utils
from mm_rag.logging_service.log_config import create_logger
from mm_rag.exceptions import ObjectUpsertionError

logger = create_logger(__name__)


class Uploader(ABC):
  def __init__(
      self,
      dynamodb: 'dynamodb.DynamoDB',
      vector_store: 'vectorstore.PineconeVectorStore',
      bucket: 's3bucket.BucketService',
  ) -> None:
    self.ddb = dynamodb
    self.vector_store = vector_store
    self.bucket = bucket

  @abstractmethod
  def upload_in_vector_store(self, file: ds.File) -> bool:
    """
    This function must be implemented in order to upsert any object in the VectorStore
    """

  @abstractmethod
  def upload_in_bucket(self, file: ds.File) -> None:
    """
    Let the subclass define how to be upserted into the s3bucket
    """

  async def aupload_in_vector_store(self, file: ds.File):
    try:
      return await asyncio.to_thread(self.upload_in_vector_store, file)

    except (AttributeError, ValueError, ObjectUpsertionError) as e:
      raise ObjectUpsertionError("PineconeVectorStore") from e

  async def aupload_in_bucket(self, file: ds.File):
    try:
      return await asyncio.to_thread(self.upload_in_bucket, file)

    except (AttributeError, ValueError, ObjectUpsertionError) as e:
      raise ObjectUpsertionError("BucketService") from e


class TxtUploader(Uploader):
  def upload_in_vector_store(self, file: ds.File) -> bool:
    try:
      logger.debug(f"upserting docs of {file.metadata.file_name} to the VectorStore")
      self.vector_store.vector_store.add_documents(file.docs)

      logger.debug("Done")

    except pinecone.PineconeException as e:
      logger.error(e)
      raise ObjectUpsertionError('PineconeVectorStore') from e 

    return True

  def upload_in_bucket(self, file: ds.File) -> None:
    logger.debug(f"Inserting {file.metadata.file_name} in {self.bucket.name}")

    try:
      self.bucket.upload_object(
        file.metadata.file_id, file.content
      )
    except ClientError as e:
      logger.error(f"Error while upserting {file.metadata.file_name} in bucket {self.bucket.name}: {e}")
      raise ObjectUpsertionError('BucketService') from e 

    logger.debug("Done")


class ImgUploader(Uploader):
  def __init__(self, dynamodb: DynamoDB, vector_store: PineconeVectorStore, bucket: BucketService) -> None:
    super().__init__(dynamodb, vector_store, bucket)

  def upload_in_vector_store(self, file: ds.File) -> bool:

    try:
      logger.debug(f'Upserting img: {file.metadata.file_name} to the VectorStore')
      self.vector_store.add_image(file.docs[0].page_content, file.metadata, file.metadata.file_id)
      logger.debug(f"Done.")

    except pinecone.PineconeException as e:
      logger.error("Error wile upserting the image %s to the VectorStore: %s" % file.metadata.file_name, e)
      raise ObjectUpsertionError('PineconeVectorStore') from e 

    return True

  def upload_in_bucket(self, file: ds.File) -> None:
    logger.debug(f"Inserting {file.metadata.file_name} in {self.bucket.name}")

    img_buffer = utils.save_img_to_buffer(file.content)

    try:
      self.bucket.upload_object_from_file(
        img_buffer,
        file.metadata.file_id
        )

    except ClientError as e:
      logger.error(f"Error while upserting {file.metadata.file_name} in bucket {self.bucket.name}: {e}")
      raise ObjectUpsertionError('BucketService') from e 

    logger.debug("Done")


class PdfUploader(Uploader):
  def __init__(self, dynamodb: DynamoDB, vector_store: PineconeVectorStore, bucket: BucketService) -> None:
    super().__init__(dynamodb, vector_store, bucket)

  def upload_in_vector_store(self, file: ds.File) -> bool:
    for i in range(len(file.content)):
      page_id = file.docs[i].id

      if not page_id:
        raise ValueError(
          f"In order to upload the given PdfFile to the VectorStore, "
          f"please provide valid ids in `docs`"
        )

      try:
        logger.debug(f"Upserting {file.metadata.file_id} to the VectorStore")
        self.vector_store.add_image(file.docs[i].page_content, file.metadata, page_id)
        logger.debug(f"Done.")

      except pinecone.PineconeException as e:
        logger.error("Error wile upserting the pdf page %d %s to the VectorStore: %s" % i, file.metadata.file_name, e)
        raise ObjectUpsertionError('PineconeVectorStore') from e 

    logger.debug(f"All pages upserted.")
    return True

  def upload_in_bucket(self, file: ds.File) -> None:
    logger.debug(f"Inserting {file.metadata.file_name} in {self.bucket.name}")

    if not file.docs:
      raise AttributeError(
        f"In order to upload PdfFile to the Bucket, please pass over the docs"
      )

    for i, page in enumerate(file.content):
      page_id = file.docs[i].id

      if not page_id:
        raise ValueError(
          f"In order to upload the given PdfFile to the VectorStore, "
          f"please provide valid ids in `docs`. Found doc {i+1} without id:\n{file.docs[i]}"
        )

      page_buffer = utils.save_img_to_buffer(page)

      try:
        self.bucket.upload_object_from_file(
          page_buffer,
          page_id
        )
      except ClientError as e:
        logger.error(f"Error while upserting {file.metadata.file_name} in bucket {self.bucket.name}: {e}")
        raise ObjectUpsertionError('BucketService') from e 

      logger.debug("Done")
