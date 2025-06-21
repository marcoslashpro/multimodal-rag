from mm_rag.pipelines.extractors import TxtExtractor, ImgExtractor, PdfExtractor, DocExtractor, Extractor
from mm_rag.pipelines.uploaders import TxtUploader, ImgUploader, PdfUploader, Uploader
import mm_rag.datastructures as ds
from mm_rag.models import dynamodb, s3bucket, vectorstore
from mm_rag.exceptions import ObjectUpsertionError
from mm_rag.logging_service.log_config import create_logger


from typing import Type
import os
import asyncio


logger = create_logger(__name__)


async def pipe(
    path: ds.Path,
    auth: ds.UserId,
    dynamo: dynamodb.DynamoDB,
    vectorstore_factory: vectorstore.VectorStoreFactory,
    bucket: s3bucket.BucketService
  ) -> None:

  _map: dict[ds.FileType, tuple[Type[Extractor], Type[Uploader]]] = {
    ds.FileType.TXT: (TxtExtractor, TxtUploader),
    ds.FileType.JPEG: (ImgExtractor, ImgUploader),
    ds.FileType.PNG: (ImgExtractor, ImgUploader),
    ds.FileType.JPG: (ImgExtractor, ImgUploader),
    ds.FileType.PDF: (PdfExtractor, PdfUploader),
    ds.FileType.DOCX: (DocExtractor, PdfUploader),
  }
  file_type = ds.FileType(os.path.splitext(path)[-1])
  vectorstore = vectorstore_factory.get_vector_store(auth)

  extractor_cls, uploader_cls = _map[file_type]
  extractor = extractor_cls()
  uploader = uploader_cls(dynamo, vectorstore, bucket)
  logger.debug(f"File type: {file_type}, piping with: {extractor=} and {uploader=}")


  file = extractor.extract(path, auth)
  logger.debug(f"Generated file: {file}")

  vector_store_upload_task = None
  bucket_upload_task = None

  try:
    async with asyncio.TaskGroup() as tg:
      logger.debug(f"Creating async upload task group")
      vector_store_upload_task = tg.create_task(uploader.aupload_in_vector_store(file))
      bucket_upload_task = tg.create_task(uploader.aupload_in_bucket(file))

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
            vectorstore.remove_object(file.metadata.file_id)
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

            bucket.remove_object(file.metadata.file_id)
            raise e
      raise e