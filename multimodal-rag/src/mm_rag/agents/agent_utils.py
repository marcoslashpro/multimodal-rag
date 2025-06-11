from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

from mm_rag.logging_service.log_config import create_logger


T = TypeVar("T", bound=BaseModel)
logger = create_logger(__name__)


class ResponseValidationError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)


class MissingResponseContentError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)


class MalformedResponseContentError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)


def validate_response(to_validate: str, parser: Type[T]) -> T:
  try:
    validated = parser.model_validate_json(to_validate)
  except ValidationError:
    raise ResponseValidationError()

  return validated