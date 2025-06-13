from typing import TYPE_CHECKING, Optional, Union
if TYPE_CHECKING:
  from mm_rag.processing.files import TxtFile, ImgFile, PdfFile
  from mm_rag.processing.handlers import ImgHandler
  from mm_rag.agents.mm_embedder import Embedder

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import os
from uuid import uuid4

from PIL import Image

from langchain_core.documents import Document


@dataclass
class Metadata:
  fileId: str
  fileName: str
  fileType: str
  created: str


class ImageTooBigException(Exception):
  def __init__(self, msg: str) -> None:
    super().__init__(msg)


class Processor(ABC):
  def __init__(
      self,
      embedder: 'Embedder',
      handler: 'ImgHandler | None' = None
  ) -> None:
    self.embedder = embedder
    if handler:
      self.handler = handler

  @abstractmethod
  def process(self, file: Union['TxtFile', 'ImgFile', 'PdfFile']) -> list[Document]:
    pass

  def gather_metadata(self, file_path: str, user_id) -> Metadata:
    fileName, fileType = os.path.splitext(os.path.basename(file_path))
    fileType = fileType.removeprefix('.')

    return Metadata(
      fileId=self.generate_id(fileType, fileName, user_id),
      fileName=fileName,
      fileType=fileType,
      created=datetime.now().isoformat(),
    )

  @abstractmethod
  def load_from_path(self, file_path: str) -> Union[str, list[Image.Image], Image.Image]:
    if not os.path.exists(file_path):
      raise FileNotFoundError(
        f"The given path {file_path} does not exist in the system."
      )

  def generate_id(self, file_type: str, file_name: str, user_id: str) -> str:
    file_name = os.path.basename(file_name)
    return f'{user_id}/{file_type}/{file_name}'

  def generate_ids(self, file_id: str, range_of_ids: int) -> list[str]:
    ids: list[str] = []

    for i in range(range_of_ids):
      ids.append(
        file_id + f'/chunk{i+1}'
      )

    return ids


class File():
  def __init__(
    self,
    file_path: str,
    owner: str,
    processor: Processor,
  ) -> None:
    self.processor = processor

    self.file_path = file_path
    self.owner = owner
    self._metadata: Optional[Metadata] = None
    self._file_content: Union[str, Image.Image, list[Image.Image], None] = None
    self._file_id: Optional[str] = None

  @property
  def metadata(self) -> Metadata:
    if not self._metadata:
      self._metadata = self.processor.gather_metadata(self.file_path, self.owner)

    return self._metadata

  @property
  def file_content(self) -> Union[str, list[Image.Image], Image.Image]:
    if not self._file_content:
      self._file_content = self.processor.load_from_path(self.file_path)

    return self._file_content

  @property
  def file_id(self) -> str:
    if not self._file_id:
      self._file_id = self.processor.generate_id(self.metadata.fileType, self.metadata.fileName, self.owner)

    return self._file_id
