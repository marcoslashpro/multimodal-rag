from mm_rag.models.dynamodb import DynamoDB
from mm_rag.models.s3bucket import BucketService
from mm_rag.models.vectorstore import PineconeVectorStore

import asyncio

from abc import ABC, abstractmethod

from botocore.exceptions import ClientError

import pinecone

import mm_rag.datastructures as ds
import mm_rag.pipelines.utils as utils
from mm_rag.logging_service.log_config import create_logger
from mm_rag.exceptions import ObjectUpsertionError, StorageError


logger = create_logger(__name__)


class Uploader(ABC):
  def __init__(
      self,
      dynamodb: DynamoDB,
      vector_store: PineconeVectorStore,
      bucket: BucketService,
  ) -> None:
    self.ddb = dynamodb
    self.vector_store = vector_store
    self.bucket = bucket

  async def aupload(self, file: ds.File) -> None:
    vector_store_upload_task = None
    bucket_upload_task = None

    try:
      async with asyncio.TaskGroup() as tg:
        logger.debug(f"Creating async upload task group")
        vector_store_upload_task = tg.create_task(self.aupload_in_vector_store(file))
        bucket_upload_task = tg.create_task(self.aupload_in_bucket(file))

    except* ObjectUpsertionError as eg:
      for e in eg.exceptions:
        if type(e) == ObjectUpsertionError:
          if e.storage == ds.Storages.BUCKET:
            if (vector_store_upload_task
                and vector_store_upload_task.done()
                and not vector_store_upload_task.cancelled()
                and not vector_store_upload_task.exception()):
              logger.warning(
                "Upload failed in Bucket, rolling back VectorStore upload."
              )
              self.vector_store.remove_object(file.metadata.file_id)
              raise e

          elif e.storage == ds.Storages.VECTORSTORE:
            if (bucket_upload_task
                and bucket_upload_task.done()
                and not bucket_upload_task.cancelled()
                and not bucket_upload_task.exception()):
              logger.warning(
                "Upload failed in VectorStore, rolling back Bucket upload."
              )

              self.bucket.remove_object(file.metadata.file_id)
              raise e
        raise e

    except* Exception as eg:
      logger.error(f"Unhandled error in exc group: {eg.exceptions}")
      raise StorageError(
        f"Something went wrong while upserting the file: {file.metadata.file_name}: {eg.exceptions}"
      )

  def upload(self, file: ds.File) -> None:
    self.upload_in_vector_store(file)
    self.upload_in_bucket(file)

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
      raise ObjectUpsertionError(ds.Storages.VECTORSTORE) from e

  async def aupload_in_bucket(self, file: ds.File):
    try:
      return await asyncio.to_thread(self.upload_in_bucket, file)

    except (AttributeError, ValueError, ObjectUpsertionError) as e:
      raise ObjectUpsertionError(ds.Storages.BUCKET) from e


class TxtUploader(Uploader):
  def upload_in_vector_store(self, file: ds.File) -> bool:
    try:
      logger.debug(f"upserting docs of {file.metadata.file_name} to the VectorStore")
      self.vector_store.add(file)

      logger.debug("Done")

    except pinecone.PineconeException as e:
      logger.error(e)
      raise ObjectUpsertionError(ds.Storages.VECTORSTORE) from e

    return True

  def upload_in_bucket(self, file: ds.File) -> None:
    logger.debug(f"Inserting {file.metadata.file_name} in {self.bucket.name}")

    try:
      for doc in file.docs:
        if doc.id is None:
          raise ObjectUpsertionError(
            storage=ds.Storages.BUCKET,
            msg=f'Missing id in doc: {doc}'
          )

        self.bucket.upload_object(
          doc.id, doc.page_content
        )
    except ClientError as e:
      raise ObjectUpsertionError(ds.Storages.BUCKET) from e

    logger.debug("Done")


class ImgUploader(Uploader):
  def __init__(self, dynamodb: DynamoDB, vector_store: PineconeVectorStore, bucket: BucketService) -> None:
    super().__init__(dynamodb, vector_store, bucket)

  def upload_in_vector_store(self, file: ds.File) -> bool:

    try:
      logger.debug(f'Upserting img: {file.metadata.file_name} to the VectorStore')
      self.vector_store.add(file)
      logger.debug(f"Done.")

    except pinecone.PineconeException as e:
      logger.error("Error wile upserting the image %s to the VectorStore: %s" % file.metadata.file_name, e)
      raise ObjectUpsertionError(ds.Storages.VECTORSTORE) from e

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
      raise ObjectUpsertionError(ds.Storages.BUCKET) from e

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
        self.vector_store.add(file)
        logger.debug(f"Done.")

      except pinecone.PineconeException as e:
        logger.error("Error wile upserting the pdf page %d %s to the VectorStore: %s" % i, file.metadata.file_name, e)
        raise ObjectUpsertionError(ds.Storages.VECTORSTORE) from e

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
        raise ObjectUpsertionError(ds.Storages.BUCKET) from e

      logger.debug("Done")


class CodeUploader(TxtUploader):
  pass