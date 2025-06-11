from typing import Any
from sqlalchemy import (
  create_engine,
  Column,
  MetaData,
  Table,
  Integer,
  String,
  Text,
  Identity,
  ForeignKey,
)
from sqlalchemy.dialects.postgresql import (
  JSONB,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import sessionmaker

from mm_rag.config.config import config


class PSQLDB:
  def __init__(
    self,
    endpoint_url: str,
    db_name: str,
    db_password: str,
    user: str = 'postgres',
    port: int = 5432,
  ) -> None:
    self.engine = create_engine(
      f"postgresql://{user}:{db_password}@"
        f"{endpoint_url}:{port}/{db_name}"
    )
    self.session = sessionmaker(
      bind=self.engine
    )()
    self.metadata_obj = MetaData()

  @property
  def users(self) -> Table:
    if not hasattr(self, "_users"):
      self._users = Table(
        "users",
        self.metadata_obj,
        Column(
          "id", Integer, Identity(always=True),
          primary_key=True
        ),
        Column(
          "username",
          String(100),
          nullable=False
        ),
        Column(
          "password",
          String(256),
          nullable=False
        )
      )
      return self._users

  @property
  def files(self) -> Table:
    if not hasattr(self, "_files"):
      self._files = Table(
        "files",
        self.metadata_obj,
        Column(
          "id", Integer, Identity(always=True),
          primary_key=True
          ),
        Column(
          "content", Text, nullable=False
        ),
        Column(
          "file_metadata", JSONB, nullable=False
        ),
        Column(
          "owned_by", Integer, ForeignKey('users.id'),
          nullable=False
        )
      )
    return self._files

  @property
  def embeddings(self) -> Table:
    if not hasattr(self, "_embeddings"):
      self._embeddings = Table(
        "embeddings",
        self.metadata_obj,
        Column(
          "id", Integer, Identity(always=True),
          primary_key=True
        ),
        Column(
          "text_content", Text, nullable=False
        ),
        Column(
          "embeddings", Vector(1024), nullable=True
        ),
        Column(
          "from_file", Integer,
          ForeignKey("files.id")
        )
      )
    return self._embeddings

  def create_all(self) -> bool:
    try:
      self.metadata_obj.create_all(
        bind=self.engine
      )
    except Exception as e:
      return False
    return True

  def create_table(self, tables: str | list[str]) -> bool:
    try:
      self.metadata_obj.create_all(
        bind=self.engine,
        tables=tables
      )
    except Exception as e:
      return False
    return True

  def delete_all(self) -> bool:
    try:
      self.metadata_obj.drop_all(
        bind=self.engine,
      )
    except Exception as e:
      return False
    return True

  def delete_table(self, tables: str | list[str]) -> bool:
    try:
      self.metadata_obj.drop_all(
        bind=self.engine,
        tables=tables
      )
    except Exception as e:
      return False
    return True

  def store_file(
      self,
      table_name: str,
      file_metadata: dict[str, Any],
      content: str | list[str],
      owned_by: int
      ) -> bool:

    if not hasattr(self, f"{table_name}"):
      raise ValueError(
        f"Table {table_name} not found"
        )

    table: Table = getattr(self, f"{table_name}")

    try:
      table.insert().values(
        file_metadata=file_metadata,
        content=content,
        owned_by=owned_by
      )
    except Exception as e:
      raise e 
    return True


# Usage
if __name__ == '__main__':
    db = PSQLDB(
        endpoint_url=config['aws']['db']['host'],
        db_name=config['aws']['db']['name'],
        db_password=config['aws']['db']['pass'],
    )
    db.delete_all()
