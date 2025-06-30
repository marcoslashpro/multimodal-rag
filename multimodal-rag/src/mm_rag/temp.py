from dataclasses import dataclass
import asyncio


from mm_rag.models.dynamodb import DynamoDB
from mm_rag.models.s3bucket import BucketService
import mm_rag.datastructures as ds
from mm_rag.exceptions import ObjectUpsertionError
from mm_rag.logging_service.log_config import create_logger


logger = create_logger(__name__)


@dataclass
class Piper:
  uploader_factory: UploaderFactory
  extractor_factory: ExtractorFactory
  vector_store_factory: VectorStoreFactory
  file_path: str
  auth: str
  dynamodb: DynamoDB
  bucket: BucketService

  def __post_init__(self) -> None:
    self.vector_store = self.vector_store_factory.get_vector_store(namespace=self.auth)
    self.extractor = self.extractor_factory.get_extractor(self.file_path)
    self.uploader = self.uploader_factory.get_uploader(
      file_path=self.file_path,
      dynamodb=self.dynamodb,
      vector_store=self.vector_store,
      bucket=self.bucket
    )

  async def pipe(self) -> None:
    file = self.extractor.extract(self.file_path, self.auth)
    vector_store_upload_task = None
    bucket_upload_task = None

    try:
      async with asyncio.TaskGroup() as tg:
        logger.debug(f"Creating async upload task group")
        vector_store_upload_task = tg.create_task(self.uploader.aupload_in_vector_store(file))
        bucket_upload_task = tg.create_task(self.uploader.aupload_in_bucket(file))

    except* ObjectUpsertionError as eg:
      for e in eg.exceptions:
        if type(e) == ObjectUpsertionError:
          if e.storage == ds.Storages.BUCKET:
            if (
              vector_store_upload_task
              and vector_store_upload_task.done()
              and not vector_store_upload_task.cancelled()
              and not vector_store_upload_task.exception()
            ):
              logger.warning(
                "Upload failed in BucketService, rolling back successful VectorStore upload."
              )
              self.vector_store.remove_object(file.metadata.file_id)
              raise e

          elif e.storage == ds.Storages.VECTORSTORE:
            if (
              bucket_upload_task
              and bucket_upload_task.done()
              and not bucket_upload_task.cancelled()
              and not bucket_upload_task.exception()
            ):
              logger.warning(
                "Upload failed in VectorStore, rolling back successful Bucket upload."
              )

              self.bucket.remove_object(file.metadata.file_id)
              raise e
        raise e
