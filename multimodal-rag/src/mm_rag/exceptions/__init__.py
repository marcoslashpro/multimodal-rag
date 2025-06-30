from typing import Literal
from mm_rag import datastructures as ds

class FileNotValidError(Exception):
    """
    This exception is raised everytime that a file with an unsupported file extension is opened.
    """


class ImageTooBigError(Exception):
    """
    This exception is raised everytime that an image is too big to be saved.
    The maximum size limit for an image is 25MB.
    """


class DocGenerationError(Exception):
    """
    This exception is raised everytime that the doc generation fails.
    """


class MissingRegionError(Exception):
  """
  This exception is particularly useful for AWS ops.
  During deployement, we will encounter some problems if we do not specify regions.
  Therefore, it is important to explain that reliably.
  """

class StorageError(Exception):
  pass

class BucketAccessError(StorageError):
  """
  Particularly useful when trying to connect to an existing bucket.
  """


class ObjectUpsertionError(StorageError):
  """
  Raise this error anytime that there is a failure in the upsertion of an object into a storage facility.
  """
  def __init__(
      self,
      storage: ds.Storages,
      msg: str | None = None
    ) -> None:
    self.storage = storage
    self.msg = msg or f'Failed to upsert into {self.storage}'
    super().__init__(self.msg)


class MissingItemError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)


class ResponseValidationError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)


class MissingResponseContentError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)


class MalformedResponseError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)


class MessageError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)


class ObjectDeletionError(StorageError):
  def __init__(self, storage: ds.Storages, *args: object) -> None:
    self.storage = storage
    super().__init__(*args)