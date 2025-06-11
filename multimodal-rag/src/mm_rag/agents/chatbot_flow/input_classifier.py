from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from mm_rag.agents.chatbot_flow import State

from mm_rag.logging_service.log_config import create_logger
from mm_rag.agents.prompts import classifier_prompt
from mm_rag.agents.agent_utils import validate_response, ResponseValidationError

from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage


logger = create_logger(__name__)


class InputClassifier(BaseModel):
  is_retrieval_required: bool = Field(
      ...,
      description="Decide if external information is required in order to "
      "properly respond to the user's query."
  )


def classify_input(state: 'State'):
  query = state['query']
  vlm = state['vlm']
  logger.debug(f"Classifying input: {query}")

  input_prompt = classifier_prompt.invoke({"query": query}).to_string().removeprefix("Human: ")

  message = BaseMessage(
    type='user',
    content=[
      {
        "type": "text",
        "text": input_prompt
      }
    ]
  )

  result = vlm.invoke([message])

  logger.debug(f"Got result of: {result}")

  validated: InputClassifier | None = None
  MAX_RETRIES = 3

  for i in range(1, MAX_RETRIES + 1):
    try:
      validated = validate_response(result.content, InputClassifier)  #type: ignore[call-args]
      if validated:
        logger.debug(f"Parsed model output: {validated}")
        logger.debug(f"Is retrieval required? {validated.is_retrieval_required}, type: {type(validated.is_retrieval_required)}")

        return {"is_retrieval_required": validated.is_retrieval_required, 'query': query}

    except ResponseValidationError as e:
      if i == MAX_RETRIES:
        return {"is_retrieval_required": True, "query": query}

      logger.info(
        f"The model failed to generate a valid classifier response after {MAX_RETRIES}"
        f"The response got is: {result.content}"
      )
      result = vlm.invoke([message])

  if not validated:
    return {"is_retrieval_required": True, "query": query}
