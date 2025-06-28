from enum import Enum
from abc import ABC, abstractmethod


from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class Code(Enum):
  PY = '.py'
  JS = '.js'


class FileTypeEnum(Enum):
  CODE = Code
  TXT = '.txt'


files = {
  FileTypeEnum.CODE.value: "CodeExtactor"
}


if __name__ == "__main__":
  print(f"{'.js' in FileTypeEnum.CODE.value}")