import os
import subprocess
from dataclasses import asdict
from abc import ABC, abstractmethod
from typing import Union

from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path

from mm_rag.exceptions import FileNotValidError, ImageTooBigError
import mm_rag.datastructures as ds
import mm_rag.pipelines.utils as utils
from mm_rag.logging_service.log_config import create_logger


logger = create_logger(__name__)


class Extractor(ABC):
    def extract(self, path: ds.Path, auth: ds.UserId) -> ds.File:
        path = validate_path(path)

        metadata = self._extract_metadata(path, auth)
        content = self._extract_content(path)
        docs = self._extract_docs(content, metadata)

        return ds.File(
            metadata=metadata,
            content=content,
            docs=docs,
        )

    @abstractmethod
    def _extract_metadata(self, path: ds.Path, auth: ds.UserId) -> ds.Metadata:
        """
        A minimal implementation of this method is already available in the abstract class.
        :param path: string, the original file path
        :param auth: string, the user that owns the file
        :return: Metadata, containing: author, file_name, file_type, created_time, and the file_id
        """
        file_name, file_type = generate_file_name_and_type(path)
        metadata = ds.Metadata(
            file_name=file_name,
            file_type=file_type.value,
            author=auth,
        )

        return metadata

    @abstractmethod
    def _extract_content(self, path: ds.Path) -> Union[str | Image.Image, list[Image.Image]]:
        pass

    @abstractmethod
    def _extract_docs(self, content: Union[str | Image.Image, list[Image.Image]], metadata: ds.Metadata) -> list[Document]:
        pass


class TxtExtractor(Extractor):
  def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

  def _extract_metadata(self, path: ds.Path, auth: ds.UserId) -> ds.Metadata:
    return super()._extract_metadata(path, auth)

  def _extract_content(self, path: ds.Path) -> str:
    with open(path, 'r', encoding='utf-8') as file:
      return file.read()

  def _extract_docs(self, content: str, metadata: ds.Metadata) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
      chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
    )

    splits = splitter.split_text(content)
    ids = utils.generate_ids(metadata.file_id, len(splits))
    docs = utils.generate_docs(ids, splits, metadata)

    return docs


class ImgExtractor(Extractor):
  def _extract_metadata(self, path: ds.Path, auth: ds.UserId) -> ds.Metadata:
    return super()._extract_metadata(path, auth)

  def _extract_content(self, path: ds.Path) -> Image.Image:
    try:
      return Image.open(path).convert("RGB")

    except Image.DecompressionBombError as e:
      logger.error(f"The given image {path} cannot be open as the size is too big.")
      raise ImageTooBigError(
        f"The given image {path} cannot be open as the size is too big."
      ) from e
    except Image.UnidentifiedImageError as e:
      logger.error(f"The given image is not an image: {path}.")
      raise FileNotValidError(
        f"The given image is not an image: {path}"
      ) from e


  def _extract_docs(self, content: Image.Image, metadata: ds.Metadata) -> list[Document]:
    processed_img = utils.process_img(content)

    docs: list[Document] = [
      Document(
        page_content=processed_img,
        metadata=asdict(metadata),
        id=metadata.file_id
      )
    ]

    return docs


class PdfExtractor(Extractor):
  def _extract_metadata(self, path: ds.Path, auth: ds.UserId) -> ds.Metadata:
    return super()._extract_metadata(path, auth)

  def _extract_content(self, path: ds.Path) -> list[Image.Image]:
    return from_pdf_path_to_pages(path)

  def _extract_docs(self, content: list[Image.Image], metadata: ds.Metadata) -> list[Document]:
    processed_pages = [utils.process_img(page) for page in content]

    ids = utils.generate_ids(metadata.file_id, len(processed_pages))
    docs = utils.generate_docs(ids, processed_pages, metadata)

    return docs


class DocExtractor(Extractor):
  def _extract_metadata(self, path: ds.Path, auth: ds.UserId) -> ds.Metadata:
    return super()._extract_metadata(path, auth)

  def _extract_content(self, path: ds.Path) -> list[Image.Image]:
    output_path, _ = os.path.splitext(path)
    output_path += '.pdf'

    convert_docx_to_pdf(path, output_path)

    return from_pdf_path_to_pages(output_path)

  def _extract_docs(self, content: list[Image.Image], metadata: ds.Metadata) -> list[Document]:
    processed_pages = [utils.process_img(page) for page in content]
    ids = utils.generate_ids(metadata.file_id, len(processed_pages))

    docs = utils.generate_docs(ids, processed_pages, metadata)

    return docs


def from_pdf_path_to_pages(path: str) -> list[Image.Image]:
  pages: list[Image.Image] = []

  for i, page in enumerate(convert_from_path(path)):
    try:
      pages.append(page.convert("RGB"))

    # TODO: Skip only one page, instead of stopping the processing
    except Image.DecompressionBombError:
      logger.error(f"Stopping pdf extraction pipeline since page {i} is too big to process")
      raise ImageTooBigError(
        f"Stopping pdf extraction pipeline since page {i} is too big to process"
      )
    except Image.UnidentifiedImageError as e:
      logger.error(f"The given image is not an image: {path}.")
      raise FileNotValidError(
        f"The given image is not an image: {path}"
      ) from e

  return pages


def convert_docx_to_pdf(input_path: str, output_path: str) -> None:
  logger.debug(f"Converting {input_path} to {output_path}")
  result = subprocess.run(
    [
      'pandoc',
      input_path,
      '-o',
      output_path,
      '--pdf-engine=tectonic'
    ],
    # !!CLOUD COMPATIBLE SETTINGS!!
    # !!TURN ON FOR DEPLOYMENT!!
    cwd='/tmp/',
    env={**os.environ, "HOME": "/tmp", "TMPDIR": "/tmp"},
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE
  )

  if result.returncode != 0:
    logger.error(
    f'Pandoc Failed while uploading {output_path}. Error: {result.stderr}, STDOUT: {result.stdout}, ErrorCode: {result.returncode}')
    raise FileNotValidError(
      f"Pandoc Failed while uploading {output_path}. Error: {result.stderr}, STDOUT: {result.stdout}, ErrorCode: {result.returncode}"
    )


def generate_file_name_and_type(file_path: str) -> tuple[str, ds.FileType]:
  file_name, _file_type = os.path.splitext(os.path.basename(file_path))

  try:
    file_type = ds.FileType(_file_type)
  except ValueError:
    raise FileNotValidError(
      f"{_file_type} is not yet a supported file type."
    )

  return file_name, file_type


def validate_path(path: str) -> str:
  if not os.path.exists(path):
    raise FileNotValidError(
      f"{path} does not lead to a file on the system"
    )
  if not os.path.isfile(path):
    raise FileNotValidError(
      f"{path} is not a file"
    )

  return path


class CodeExtractor(Extractor):
  pass