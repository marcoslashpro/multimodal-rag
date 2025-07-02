from PIL import Image
from langchain_core.documents import Document
from typing import Callable, TypeAlias, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum



class Code(Enum):
  CPP = '.cpp'
  CSHARP = '.cs'
  GO = '.go'
  HTML = '.html'
  JAVA = '.java'
  JS = '.js'
  KT = '.kt'
  LUA = '.lua'
  MD = '.md'
  PHP = '.php'
  PY = '.py'
  RB = '.rb'
  RS = '.rs'
  SCALA = '.scala'
  SWIFT = '.swift'
  TEX = '.tex'
  TS = '.ts'


class Img(Enum):
  JPEG = '.jpeg'
  PNG = '.png'
  JPG = '.jpg'


class Audio(Enum):
  MP3 = '.mp3'
  WAV = '.wav'


class FileType(Enum):
  IMAGE = Img
  PDF = '.pdf'
  DOCX = '.docx'
  TXT = '.txt'
  CODE = Code
  AUDIO = Audio


@dataclass
class Metadata:
  file_name: str
  file_type: str
  author: str
  created: str = field(default=datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

  def __post_init__(self):
    self.file_id = f'{self.author}/{self.file_type}/{self.file_name}'
    self.collection = 'audio' if self.file_type in FileType.AUDIO.value else 'other'


@dataclass
class File:
  metadata: Metadata
  content: str | Image.Image | list[Image.Image]
  docs: list[Document]
  embeddings: list[list[float]]


Path: TypeAlias = str
UserId: TypeAlias = str
EmbeddingFunc = Callable[[Union[str]], list[float]]


class Storages(Enum):
  BUCKET = 'BucketService'
  VECTORSTORE = 'PineconeVectorStore'
  DYNAMO = 'DynamoDB'


class Collection(Enum):
  AUDIO = '/audio'
  OTHER = '/other'
