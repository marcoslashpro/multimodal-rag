from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

from mm_rag.logging_service.log_config import create_logger
from mm_rag.exceptions import ResponseValidationError


T = TypeVar("T", bound=BaseModel)
logger = create_logger(__name__)


def validate_response(to_validate: str, parser: Type[T]) -> T:
  try:
    validated = parser.model_validate_json(to_validate)
  except ValidationError:
    raise ResponseValidationError()

  return validated