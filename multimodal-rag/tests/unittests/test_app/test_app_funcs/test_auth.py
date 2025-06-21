from mm_rag.api.utils import authorize
from mm_rag.models.dynamodb import DynamoDB
from mm_rag.exceptions import MissingItemError

from fastapi import HTTPException

from unittest.mock import MagicMock, patch

import unittest


class TestAuth(unittest.TestCase):
  def setUp(self):
    self.mock_ddb = MagicMock(DynamoDB)
    self.test_pat = 'testPAT'
    self.test_id = 'testId'

  def test_successfull_query(self):
    self.mock_ddb.query_with_gsi.return_value = [{
      'userId': self.test_id,
      'PAT': self.test_pat
    }]

    auth_user = authorize(self.mock_ddb, self.test_pat)

    self.assertEqual(
      auth_user.pat, self.test_pat
    )
    self.assertEqual(
      auth_user.user_id, self.test_id
    )

  def test_missing_item_query_raises(self):
    self.mock_ddb.get_from_table.side_effect = MissingItemError

    with self.assertRaises(HTTPException):
      authorize(self.mock_ddb, self.test_id)

  def test_malformed_user_raises(self):
    malformed_users = [
      {
        "userId": self.test_id,
        "NoPat": "Missing PAT token"
      },
      {
        "noUserId": "Missing userId",
        "PAT": self.test_pat
      }
    ]

    for user in malformed_users:
      with self.subTest(user=user):
        self.mock_ddb.get_from_table.return_value = user
        with self.assertRaises(HTTPException):
          authorize(self.mock_ddb, 'whatever')

if __name__ == "__main__":
  unittest.main()