import mm_rag.pipelines.extractors as extr
import mm_rag.pipelines.uploaders as upl
import mm_rag.pipelines.retrievers as retr
import mm_rag.datastructures as ds
from mm_rag.models import dynamodb, s3bucket, vectorstore as vs
from mm_rag.exceptions import FileNotValidError
from mm_rag.agents.mm_embedder import Embedder
from mm_rag.logging_service.log_config import create_logger

from typing import Type
import os
import asyncio
from dataclasses import dataclass


logger = create_logger(__name__)


class ComponentFactory:
  def __init__(
    self,
    embedder: Embedder,
    api_key: str,
    index_name: str,
    cloud: str,
    region: str,
    dynamodb: dynamodb.DynamoDB,
    bucket: s3bucket.BucketService,
  ) -> None:
    self.dynamodb = dynamodb
    self.bucket = bucket
    self.embedder = embedder
    self.api_key = api_key
    self.index_name = index_name
    self.cloud = cloud
    self.region = region

  def get_extractor(self, file_path: ds.Path, auth: ds.UserId) -> extr.Extractor:
    file_ext = self.get_file_ext(file_path)
    if not hasattr(self, 'vector_store'):
      self.vector_store = self.get_vector_store(auth)

    file_ext = os.path.splitext(file_path)[-1]
    if file_ext == ds.FileType.TXT.value:
      return extr.TxtExtractor(self.embedder.embed_query)
    if file_ext in ds.FileType.IMAGE.value:
      return extr.ImgExtractor(self.embedder.embed_img)
    if file_ext == ds.FileType.PDF.value:
      return extr.PdfExtractor(self.embedder.embed_img)
    if file_ext == ds.FileType.DOCX.value:
      return extr.DocExtractor(self.embedder.embed_img)
    elif file_ext in ds.FileType.CODE.value:
      return extr.CodeExtractor(self.embedder.embed_query)

    raise FileNotValidError(
      f"File type: {file_ext} not yet supported"
    )

  def get_uploader(self, file_path: ds.Path, auth: ds.UserId) -> upl.Uploader:
    file_ext = self.get_file_ext(file_path)
    if not hasattr(self, 'vector_store'):
      self.vector_store = self.get_vector_store(auth)

    if file_ext == ds.FileType.TXT.value:
      return upl.TxtUploader(
        dynamodb=self.dynamodb,
        vector_store=self.vector_store,
        bucket=self.bucket
    )
    if file_ext in ds.FileType.IMAGE.value:
      return upl.ImgUploader(
        dynamodb=self.dynamodb,
        vector_store=self.vector_store,
        bucket=self.bucket
    )
    if (file_ext == ds.FileType.PDF.value or
        file_ext == ds.FileType.DOCX.value):
      return upl.PdfUploader(
        dynamodb=self.dynamodb,
        vector_store=self.vector_store,
        bucket=self.bucket
    )
    elif file_ext in ds.FileType.CODE.value:
      return upl.CodeUploader(
        dynamodb=self.dynamodb,
        vector_store=self.vector_store,
        bucket=self.bucket
    )

    raise FileNotValidError(
      f"File type: {file_ext} not yet supported"
    )

  def get_vector_store(self, auth: ds.UserId) -> vs.PineconeVectorStore:
    return vs.PineconeVectorStore(
      embedder=self.embedder,
      api_key=self.api_key,
      index_name=self.index_name,
      namespace=auth,
      cloud=self.cloud,
      region=self.region
    )

  def get_retriever(self, auth: ds.UserId, top_k: int = 3) -> retr.Retriever:
    if not hasattr(self, 'vector_store'):
      self.vector_store = self.get_vector_store(auth)

    return retr.Retriever(
      self.vector_store,
      self.dynamodb,
      self.bucket,
      self.embedder,
      top_k
    )

  @staticmethod
  def get_file_ext(path: ds.Path) -> str:
    return os.path.splitext(path)[-1]


class Piper:
  def __init__(
    self,
    factory: ComponentFactory
  ) -> None:
    self.factory = factory

  def _get(self, file_path, auth) -> tuple[upl.Uploader, extr.Extractor]:  # Add `embedder: Embedder`
    extractor = self.factory.get_extractor(file_path, auth)  # Add embedder
    uploader = self.factory.get_uploader(file_path, auth)

    return uploader, extractor

  async def pipe(self, file_path: ds.Path, auth: ds.UserId) -> None:  
    uploader, extractor = self._get(file_path, auth)
    file = extractor.extract(file_path, auth)
    await uploader.aupload(file)
