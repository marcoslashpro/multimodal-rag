from PIL import Image
from langchain_core.documents import Document
from typing import Callable, TypeAlias, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class Code(Enum):
  BASH = '.bash'
  CPP = '.cpp'
  CSHARP = '.cs'
  CSS = '.css'
  DOCKERFILE = 'Dockerfile'
  GO = '.go'
  HTML = '.html'
  JAVA = '.java'
  JS = '.js'
  JSON = '.json'
  KT = '.kt'
  LUA = '.lua'
  MAKEFILE = 'Makefile'
  MD = '.md'
  PHP = '.php'
  PY = '.py'
  R = '.r'
  RB = '.rb'
  RS = '.rs'
  SCALA = '.scala'
  SH = '.sh'
  SQL = '.sql'
  SWIFT = '.swift'
  TEX = '.tex'
  TS = '.ts'
  XML = '.xml'
  YAML = '.yaml'


class Img(Enum):
  JPEG = '.jpeg'
  PNG = '.png'
  JPG = '.jpg'


class FileType(Enum):
  IMAGE = Img
  PDF = '.pdf'
  DOCX = '.docx'
  TXT = '.txt'
  CODE = Code


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


class Storages(Enum):
  BUCKET = 'BucketService'
  VECTORSTORE = 'PineconeVectorStore'
  DYNAMO = 'DynamoDB'