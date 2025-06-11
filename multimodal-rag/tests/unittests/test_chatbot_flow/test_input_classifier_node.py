import unittest
from unittest.mock import patch, MagicMock

from mm_rag.agents.chatbot_flow.input_classifier import classify_input, InputClassifier, ResponseValidationError

class TestClassifyInput(unittest.TestCase):
    def setUp(self):
        self.state = {
            "query": "Do I need external info?",
            "vlm": MagicMock()
        }

    @patch("mm_rag.agents.chatbot_flow.input_classifier.classifier_prompt")
    @patch("mm_rag.agents.chatbot_flow.input_classifier.validate_response")
    def test_valid_response(self, mock_validate, mock_prompt):
        # Mock prompt and VLM
        mock_prompt.invoke.return_value.to_string.return_value = "Human: Should I retrieve?"
        self.state["vlm"].invoke.return_value.content = '{"is_retrieval_required": "yes"}'
        mock_validate.return_value = InputClassifier(is_retrieval_required=True)

        result = classify_input(self.state)
        self.assertEqual(result, {"is_retrieval_required": True, "query": "Do I need external info?"})

    @patch("mm_rag.agents.chatbot_flow.input_classifier.classifier_prompt")
    @patch("mm_rag.agents.chatbot_flow.input_classifier.validate_response")
    def test_retry_on_response_validation_error(self, mock_validate, mock_prompt):
        # First call raises, second call succeeds
        mock_prompt.invoke.return_value.to_string.return_value = "Human: Should I retrieve?"
        self.state["vlm"].invoke.return_value.content = '{"is_retrieval_required": "no"}'
        mock_validate.side_effect = [ResponseValidationError("bad"), InputClassifier(is_retrieval_required=False)]

        result = classify_input(self.state)
        self.assertEqual(result, {"is_retrieval_required": False, "query": "Do I need external info?"})
        self.assertEqual(mock_validate.call_count, 2)

    @patch("mm_rag.agents.chatbot_flow.input_classifier.classifier_prompt")
    @patch("mm_rag.agents.chatbot_flow.input_classifier.validate_response")
    def test_exhaust_retries(self, mock_validate, mock_prompt):
        mock_prompt.invoke.return_value.to_string.return_value = "Human: Should I retrieve?"
        self.state["vlm"].invoke.return_value.content = '{"is_retrieval_required": "maybe"}'
        mock_validate.side_effect = ResponseValidationError("bad")

        result = classify_input(self.state)
        self.assertEqual(result, {"is_retrieval_required": True, "query": "Do I need external info?"})
        self.assertEqual(mock_validate.call_count, 3)


if __name__ == "__main__":
    unittest.main()