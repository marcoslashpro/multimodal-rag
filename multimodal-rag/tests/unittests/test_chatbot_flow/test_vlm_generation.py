from mm_rag.agents.vlm import VLM, MissingResponseContentError
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from huggingface_hub import InferenceClient, ChatCompletionOutput

from unittest.mock import MagicMock, patch
import unittest

class TestAgentGeneration(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock(InferenceClient)
        self.mock_client.model = 'mock model'
        self.vlm = VLM(
            model=self.mock_client
        )
        self.mock_output = ChatCompletionOutput(
            id="mock_id",
            created=1234567890,
            model="Qwen2.5-7B-Instruct",
            system_fingerprint="mock-fingerprint",
            choices=[
                MagicMock(
                    index=0,
                    message=MagicMock(
                        role="assistant",
                        content="This is a mocked response."
                    ),
                    finish_reason="stop"
                )
            ],
            usage=None
        )

    def test_model_generate_args(self):
        expected_args = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "test text"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "mock image url"
                        }
                    }
                ]
            }
        ]

        with patch.object(self.vlm.model.chat.completions, 'create', return_value=self.mock_output) as mock_completion:
            messages = [
                BaseMessage(
                    type='human',
                    content=[
                        {
                            "type": "text", "text": "test text"
                        },
                        {
                            "type": "image_url", "image_url": {"url": "mock image url"}
                        }
                    ]
                )
            ]

            self.vlm._generate(messages)
            mock_completion.assert_called_once_with(
                messages=expected_args,
                model=self.vlm.model.model
            )

    def test_missing_content_raises(self):
        # Simulate a response with None content
        mock_output = ChatCompletionOutput(
            id="mock_id",
            created=1234567890,
            model="Qwen2.5-7B-Instruct",
            system_fingerprint="mock-fingerprint",
            choices=[
                MagicMock(
                    index=0,
                    message=MagicMock(
                        role="assistant",
                        content=None
                    ),
                    finish_reason="stop"
                )
            ],
            usage=None
        )
        with patch.object(self.vlm.model.chat.completions, 'create', return_value=mock_output):
            messages = [
                BaseMessage(
                    type='human',
                    content=[
                        {
                            "type": "text", "text": "test text"
                        }
                    ]
                )
            ]
            with self.assertRaises(MissingResponseContentError):
                self.vlm._generate(messages)

    def test_chatresult_formatting_error(self):
        # Simulate a response with content that causes AIMessage to raise
        mock_output = ChatCompletionOutput(
            id="mock_id",
            created=1234567890,
            model="Qwen2.5-7B-Instruct",
            system_fingerprint="mock-fingerprint",
            choices=[
                MagicMock(
                    index=0,
                    message=MagicMock(
                        role="assistant",
                        content={"not": "a string"}  # AIMessage expects a string
                    ),
                    finish_reason="stop"
                )
            ],
            usage=None
        )
        with patch.object(self.vlm.model.chat.completions, 'create', return_value=mock_output):
            messages = [
                BaseMessage(
                    type='human',
                    content=[
                        {
                            "type": "text", "text": "test text"
                        }
                    ]
                )
            ]
            with self.assertRaises(Exception):
                self.vlm._generate(messages)

    def test_generate_with_humanmessage(self):
        # Accepts HumanMessage as input
        with patch.object(self.vlm.model.chat.completions, 'create', return_value=self.mock_output) as mock_completion:
            messages = [
                HumanMessage(
                    content=[
                        {
                            "type": "text", "text": "test text"
                        }
                    ]
                )
            ]
            self.vlm._generate(messages)
            mock_completion.assert_called_once()

if __name__ == "__main__":
    unittest.main()