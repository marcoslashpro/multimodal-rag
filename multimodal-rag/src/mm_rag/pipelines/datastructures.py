from PIL import Image
from langchain_core.documents import Document
from typing import Callable, TypeAlias, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class FileType(Enum):
    JPEG = '.jpeg'
    PNG = '.png'
    JPG = '.jpg'
    PDF = '.pdf'
    DOCX = '.docx'
    TXT = '.txt'


@dataclass
class Metadata:
  file_name: str
  file_type: str
  author: str
  created: str = field(default=datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

  def __post_init__(self):
      self.file_id = f'{self.author}/{self.file_type}/{self.file_name}'


@dataclass
class File:
    metadata: Metadata
    content: str | Image.Image | list[Image.Image]
    docs: list[Document]


Path: TypeAlias = str
UserId: TypeAlias = str
