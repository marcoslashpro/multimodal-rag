import os
from typing import Union

from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mm_rag.processing.base import Processor, ImageTooBigException
from mm_rag.logging_service.log_config import create_logger
from mm_rag.processing.files import TxtFile, ImgFile, PdfFile
from mm_rag.processing.handlers import ImgHandler
from mm_rag.agents.mm_embedder import Embedder


from pdf2image import convert_from_path


logger = create_logger(__name__)


class ProcessorFactory:
  def get_processor(
      self,
      file_path: str,
      embedder: 'Embedder',
      img_handler: 'ImgHandler'
      ) -> Union['TxtProcessor', 'PdfProcessor', 'ImgProcessor']:
    processor = self.create_processor(file_path, embedder, img_handler)
    return processor

  def create_processor(
      self,
      file_path: str,
      embedder: 'Embedder',
      img_handler: 'ImgHandler'
      ) -> Union['TxtProcessor', 'PdfProcessor', 'ImgProcessor']:
    _, file_type = os.path.splitext(file_path)

    if file_type == '.txt':
      return TxtProcessor(embedder, img_handler)

    elif file_type == '.pdf':
      return PdfProcessor(embedder, img_handler)

    elif file_type in ['.png', '.jpeg', '.jpg']:
      return ImgProcessor(embedder, img_handler)

    raise ValueError(
      f'File type {file_type} not yet supported.'
    )



class TxtProcessor(Processor):
  def load_from_path(self, file_path: str) -> str:
    super().load_from_path(file_path)

    with open(file_path, 'r') as f:
      return f.read()

  def process(self, file: Union[TxtFile, ImgFile, PdfFile], chunk_size: int = 500, chunk_overlap: int = 200) -> list[Document]:
    msg = f"Expected type TxtFile for processing, got {type(file)}"

    if not isinstance(file, TxtFile):
      raise ValueError(msg)

    logger.debug(f"Instantiating the splitter.")
    splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    logger.debug(f"Generating the docs for file: {file.file_path}")
    splits = splitter.split_text(file.file_content)
    ids = self.generate_ids(file.file_id, len(splits))

    docs: list[Document] = []

    for id, split in zip(ids, splits):
      docs.append(
        Document(
          page_content=split,
          metadata=file.metadata.__dict__,
          id=id
        )
      )

    return docs


class ImgProcessor(Processor):
  def __init__(
      self,
      embedder: Embedder,
      handler: ImgHandler | None = None
    ) -> None:
    super().__init__(embedder, handler)
    if not self.handler:
      raise RuntimeError(
        f"Please provide a ImgHandler for ImgProcessor"
      )

  def load_from_path(self, file_path: str) -> Image.Image:
    super().load_from_path(file_path)

    try:
      return Image.open(file_path).convert("RGB")

    except Image.DecompressionBombError as e:
      raise ImageTooBigException(
        f"The given image {file_path} cannot be open as the size is too big."
      )

  def process(self, file: Union[TxtFile, ImgFile, PdfFile]) -> list[Document]:
    msg = f"Expected type PdfFile for processing, got {type(file)}"

    if not isinstance(file, ImgFile):
      raise ValueError(msg)

    logger.debug(f"Adjusting orientation for Img: {file.file_path}")
    processed_img: Image.Image = self.handler.adjust_orientation(file.file_content)

    logger.debug(f"Adjusting shape for image: {file.file_path}")
    processed_img = self.handler.adjust_shape(processed_img)

    encoded_img = self.handler.base64_encode(processed_img)

    docs: list[Document] = [
      Document(
        page_content=encoded_img,
        metadata=file.metadata.__dict__,
        id=file.file_id
      )
    ]

    return docs


class PdfProcessor(Processor):
  def __init__(self, embedder: Embedder, handler: ImgHandler | None = None) -> None:
    super().__init__(embedder, handler)
    if not self.handler:
      raise RuntimeError(
        f"Please provide an ImgHandler for PdfProcessor"
      )

  def load_from_path(self, file_path: str) -> list[Image.Image]:
    super().load_from_path(file_path)

    pages: list[Image.Image] = []

    for i, page in enumerate(convert_from_path(file_path)):
      try:
        pages.append(page.convert("RGB"))

      # TODO: Skip only one page, instead of stopping the processing
      except Image.DecompressionBombError:
        raise ImageTooBigException(
          f"Skipping page: {i}, found {page = } too big to process."
        )
    return pages

  def process(self, file: Union[ImgFile, TxtFile, PdfFile]) -> list[Document]:
    if not isinstance(file, PdfFile):
      raise ValueError(
        f'Expected file of type PdfFile, got {type(file)}'
      )

    processed_pages: list[str] = []

    for page in file.file_content:
      processed_page = self.handler.adjust_orientation(page)
      processed_page = self.handler.adjust_shape(processed_page)
      encoded_page: str = self.handler.base64_encode(processed_page)

      processed_pages.append(encoded_page)

    ids = self.generate_ids(file.file_id, len(processed_pages))

    docs: list[Document] = []

    for id, page in zip(ids, processed_pages):
      docs.append(
        Document(
          page_content=page,
          metadata=file.metadata.__dict__,
          id=id
        )
      )

    return docs