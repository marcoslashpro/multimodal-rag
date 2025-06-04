from smolagents import InferenceClientModel, CodeAgent, DuckDuckGoSearchTool
from smolagents.tools import Tool
from huggingface_hub.errors import HfHubHTTPError

class VLM:
  def __init__(
      self,
      tools: list[Tool],
      model_id: str = "google/gemma-3-27b-it",
  ) -> None:
    self.tools = tools
    self.model_id = model_id
    self._agent: CodeAgent | None = None

  @property
  def agent(self) -> CodeAgent:
    if not self._agent:
      try:
        agent = CodeAgent(
          model=InferenceClientModel(),
          tools=self.tools,
          add_base_tools=True
        )
      except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 401:
          print('In order to use the code agent, you must provide a valid '
          'HF_TOKEN(With READ permission and permissions to `Make calls to Inference Providers`)' \
          ' by doing `huggingface-cli login` and then inserting your token.')
          raise
        raise e

      return agent

    return self._agent
  
  def chat(self, prompt: str) -> None:
    print(self.agent(prompt))