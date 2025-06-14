from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
  from mm_rag.processing.processors import PdfProcessor, ImgProcessor, TxtProcessor, DocxProcessor
  from mm_rag.processing.base import Metadata

from mm_rag.processing.base import File, Processor
from mm_rag.logging_service.log_config import create_logger

import os

from PIL import Image


logger = create_logger(__name__)


class FileFactory:
  def get_file(
      self,
      file_path: str,
      owner: str,
      processor: Union['PdfProcessor', 'ImgProcessor', 'TxtProcessor', 'DocxProcessor']
    ) -> Union['TxtFile', 'ImgFile', 'PdfFile', 'DocxFile']:
    file = self.create_file(file_path, owner, processor)
    return file

  def create_file(
      self,
      file_path: str,
      owner: str,
      processor: Union['PdfProcessor', 'ImgProcessor', 'TxtProcessor', 'DocxProcessor']
    ) -> Union['TxtFile', 'ImgFile', 'PdfFile', 'DocxFile']:
    _, file_ext = os.path.splitext(file_path)

    if file_ext == '.txt':
      return TxtFile(file_path, owner, processor)

    if file_ext == '.pdf':
      return PdfFile(file_path, owner, processor)

    if file_ext in ['.jpeg', '.jpg', '.png']:
      return ImgFile(file_path, owner, processor)

    if file_ext == '.docx':
      return DocxFile(file_path, owner, processor)

    else:
      raise ValueError(
        f'File type {file_ext} not yet supported'
      )


class TxtFile(File):
  @property
  def file_content(self) -> str:
    self._file_content = super().file_content

    if not isinstance(self._file_content, str):
      raise ValueError(
        f"Expected file_content to be of type str, got {type(self._file_content)}"
      )

    return self._file_content


class ImgFile(File):

  _encodings: str | None = None

  @property
  def file_content(self) -> Image.Image:
    self._file_content = super().file_content

    if not isinstance(self._file_content, Image.Image):
      raise ValueError(f'Expected file_content of type Image.Image, got {type(self._file_content)}')

    return self._file_content

  @property
  def encodings(self) -> str:
    return self.processor.handler.base64_encode(self.file_content)


class PdfFile(File):

  _encodings: list[str] | None = None

  @property
  def file_content(self) -> list[Image.Image]:
    self._file_content = super().file_content

    if not isinstance(self._file_content, list):
      raise ValueError(
        f"Expected file_content of type list[Image.Image], got {type(self._file_content)}"
      )

    for page in self._file_content:
      if not isinstance(page, Image.Image):
        raise ValueError(
          f"Expected all pages to be of type Image.Image, got pages {type(page)}"
        )

    return self._file_content

  @property
  def encodings(self) -> list[str]:
    _encodings: list[str] = []

    for page in self.file_content:
      _encodings.append(self.processor.handler.base64_encode(page))

    return _encodings


class DocxFile(PdfFile):
  def __init__(self, file_path: str, owner: str, processor: Processor) -> None:
    super().__init__(file_path, owner, processor)