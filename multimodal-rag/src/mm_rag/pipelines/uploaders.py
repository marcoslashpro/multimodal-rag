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

    except* ObjectUpsertionError as upsertion_eg:
      if (vector_store_upload_task
          and vector_store_upload_task.done()
          and not vector_store_upload_task.cancelled()
          and not vector_store_upload_task.exception()):
        logger.warning(
          "Upload failed in Bucket, rolling back VectorStore upload."
        )
        self.vector_store.remove_object(file.metadata.file_id)

      if (bucket_upload_task
          and bucket_upload_task.done()
          and not bucket_upload_task.cancelled()
          and not bucket_upload_task.exception()):
          logger.warning(
            "Upload failed in VectorStore, rolling back Bucket upload."
          )
          self.bucket.remove_object(file.metadata.file_id)

    except* Exception as eg:
      raise StorageError(
        f'Unexpected error while uploading: {eg.exceptions}'
      )

  def upload(self, file: ds.File) -> None:
    self.upload_in_vector_store(file)
    self.upload_in_bucket(file)

  @abstractmethod
  def upload_in_vector_store(self, file: ds.File) -> bool:
    """
    This function must be implemented in order to upsert any object in the VectorStore
    """
    logger.info(f"Upserting file: {file.metadata.file_name} into {self.vector_store.namespace}")
    if not file.docs:
      raise ObjectUpsertionError(
        storage=ds.Storages.VECTORSTORE,
        msg=f"Malformed file, missing docs: {file}"
      )

    if not len(file.docs) == len(file.embeddings):
      raise ObjectUpsertionError(
        storage=ds.Storages.VECTORSTORE,
        msg=f"Length of docs and embeddings of file {file.metadata.file_id} do not match: "
          f"{len(file.docs)} != {len(file.embeddings)}"
      )

    if not file.docs or not file.embeddings:
      raise ObjectUpsertionError(
        storage=ds.Storages.VECTORSTORE,
        msg=f'Malformed file: {file}'
      )

    if not all([doc.id for doc in file.docs]):
      raise ObjectUpsertionError(
        storage=ds.Storages.VECTORSTORE,
        msg=f"Malformed file with missing ids: {file}"
      )

  @abstractmethod
  def upload_in_bucket(self, file: ds.File) -> bool:
    """
    Let the subclass define how to be upserted into the s3bucket
    """
    logger.debug(f"Inserting {file.metadata.file_name} in {self.bucket.name}")
    if not file.docs:
      raise ObjectUpsertionError(
        storage=ds.Storages.BUCKET,
        msg=f"Malformed file, missing docs: {file}"
      )

    if not all([doc.id for doc in file.docs]):
      raise ObjectUpsertionError(
        storage=ds.Storages.BUCKET,
        msg=f"Malformed file with missing ids: {file}"
      )

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

  def _generate_full_namespace(self, auth: ds.UserId, collection: str) -> str:
    return auth + f"/{collection}"

class TxtUploader(Uploader):
  def upload_in_vector_store(self, file: ds.File) -> bool:
    super().upload_in_vector_store(file)

    for doc, embeddings in zip(file.docs, file.embeddings):
      self.vector_store.upload(
        id=doc.id,  # type: ignore[already-checked]
        embeddings=embeddings,
        metadata=doc.metadata,
        collection=self._generate_full_namespace(file.metadata.author, file.metadata.collection)
      )

    return True

  def upload_in_bucket(self, file: ds.File) -> bool:
    super().upload_in_bucket(file)

    for doc in file.docs:
      self.bucket.upload_object(
        doc.id, doc.page_content  # type: ignore[already-checked]
      )

    return True


class ImgUploader(Uploader):
  def __init__(self, dynamodb: DynamoDB, vector_store: PineconeVectorStore, bucket: BucketService) -> None:
    super().__init__(dynamodb, vector_store, bucket)

  def upload_in_vector_store(self, file: ds.File) -> bool:
    super().upload_in_vector_store(file)

    for doc, embeddings in zip(file.docs, file.embeddings):
      self.vector_store.upload(
        id=doc.id,  # type: ignore[already-checked]
        embeddings=embeddings,
        metadata=doc.metadata,
        collection=self._generate_full_namespace(file.metadata.author, file.metadata.collection)
      )

    return True

  def upload_in_bucket(self, file: ds.File) -> bool:
    super().upload_in_bucket(file)

    img_buffer = utils.save_img_to_buffer(file.content)

    self.bucket.upload_object_from_file(
      img_buffer,
      file.metadata.file_id
    )

    return True


class PdfUploader(Uploader):
  def __init__(self, dynamodb: DynamoDB, vector_store: PineconeVectorStore, bucket: BucketService) -> None:
    super().__init__(dynamodb, vector_store, bucket)

  def upload_in_vector_store(self, file: ds.File) -> bool:
    super().upload_in_vector_store(file)

    if not len(file.content) == len(file.docs):
      raise ObjectUpsertionError(
        storage=ds.Storages.BUCKET,
        msg=f"Malformed file, found mismatch between length of content and docs."
        f"{len(file.content)} != {len(file.docs)}, full file: {file}"
      )

    for i in range(len(file.content)):
      page = file.docs[i]
      embeddings = file.embeddings[i]

      self.vector_store.upload(
        id=page.id,  # type: ignore[already-checked]
        embeddings=embeddings,
        metadata=page.metadata,
        collection=self._generate_full_namespace(file.metadata.author, file.metadata.collection)
      )

    logger.debug(f"All pages upserted.")
    return True

  def upload_in_bucket(self, file: ds.File) -> bool:
    super().upload_in_bucket(file)

    for i, page in enumerate(file.content):
      page_id = file.docs[i].id

      page_buffer = utils.save_img_to_buffer(page)

      self.bucket.upload_object_from_file(
        page_buffer,
        page_id  # type: ignore[already-checked]
      )

    return True


class CodeUploader(TxtUploader):
  pass