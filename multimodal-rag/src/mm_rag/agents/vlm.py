from huggingface_hub import InferenceClient
from typing import Any, List, Literal, Optional
from dataclasses import dataclass

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage
)

from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from mm_rag.logging_service.log_config import create_logger
from mm_rag.exceptions import MissingResponseContentError


logger = create_logger(__name__)

@dataclass
class Content:
  type: Literal['text', 'image_url']
  image_url: Optional[dict[Literal['url'], str]] = None
  text: Optional[str] = None


class VLM(BaseChatModel):
  """
  This is a multimodal implementation of a Langchain BaseChatModel.
  The scope of this VLM is to be implemented in a LangGraph in order to provide
  augmented responses to the user's queries about images, various text files and also pdfs.
  """
  # To make this work we'll use start by taking advantage of hugging_face InferenceClient
  model: InferenceClient = Field(alias='model')

  # This other params are optional
  max_tokens: Optional[int] = None
  temperature: Optional[float] = None
  top_p: Optional[float] = None
  top_k: Optional[float] = None

  def _generate(self, messages: List[BaseMessage], stop: List[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
    final_message = [
      {
        "role": "user",
        "content": [message.content for message in messages][0]
      }
    ]

    logger.debug(f"Invoking model {self.model.model} with: {final_message}")

    response = self.model.chat.completions.create(
      messages=final_message,
      model=self.model.model
    )

    logger.debug(f"Full response: {response}")
    logger.debug(f"Full completion: {response.choices[0]}")

    content = response.choices[0].message.content
    if content is None:
      logger.error(f"Was not able to generate response content for the given query {messages}")
      raise MissingResponseContentError(
        f"Response object for query: {messages} has no content key, full response: {response}"
      )

    # Format the response in a LangChain-friendly way
    message = AIMessage(content=content)
    generation = ChatGeneration(message=message)
    result = ChatResult(generations=[generation])

    return result

  @property
  def _llm_type(self) -> str:
    return 'qwen-vlm'

if __name__ == '__main__':
  pass
