import unittest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError

from mm_rag.models.dynamodb import DynamoDB
from mm_rag.exceptions import MissingItemError

class TestDynamoDB(unittest.TestCase):
    def setUp(self):
        self.dynamodb = DynamoDB()
        self.dynamodb.ddb = MagicMock()
        self.mock_table = MagicMock()
        # Patch _validate_table to always return the mock table
        self.dynamodb._validate_table = MagicMock(return_value=self.mock_table)

    def test_add_user_success(self):
        self.mock_table.put_item.return_value = None
        result = self.dynamodb.add_user("users", "pat123", 'testId')
        self.assertTrue(result)
        self.mock_table.put_item.assert_called_once()

    def test_add_user_client_error(self):
        self.mock_table.put_item.side_effect = ClientError(
            error_response={"Error": {"Code": "400", "Message": "Bad Request"}},
            operation_name="PutItem"
        )
        with self.assertRaises(ClientError):
            result = self.dynamodb.add_user("users", "pat123", 'testId')
            self.assertFalse(result)

    def test_store_file_success(self):
        self.mock_table.put_item.return_value = None
        result = self.dynamodb.store_file("files", "file1", "user1", {"meta": "data"})
        self.assertTrue(result)
        self.mock_table.put_item.assert_called_once()

    def test_store_file_client_error(self):
        self.mock_table.put_item.side_effect = ClientError(
            error_response={"Error": {"Code": "400", "Message": "Bad Request"}},
            operation_name="PutItem"
        )
        result = self.dynamodb.store_file("files", "file1", "user1", {"meta": "data"})
        self.assertFalse(result)

    def test_get_from_table_success(self):
        self.mock_table.get_item.return_value = {"Item": {"userId": "user1", "fileId": "file1"}}
        result = self.dynamodb.get_from_table("files", {"userId": "user1", "fileId": "file1"})
        self.assertEqual(result, {"userId": "user1", "fileId": "file1"})

    def test_get_from_table_missing_item(self):
        self.mock_table.get_item.return_value = {}
        with self.assertRaises(MissingItemError):
            self.dynamodb.get_from_table("files", {"userId": "user1", "fileId": "file1"})

    def test_get_from_table_client_error(self):
        self.mock_table.get_item.side_effect = ClientError(
            error_response={"Error": {"Code": "400", "Message": "Bad Request"}},
            operation_name="GetItem"
        )
        with self.assertRaises(ClientError):
            self.dynamodb.get_from_table("files", {"userId": "user1", "fileId": "file1"})

    def test_update_file_from_table_success(self):
        self.mock_table.update_item.return_value = None
        update_expression = {
            "expression": "SET s3Path = :val1",
            "value": {":val1": "new/path"}
        }
        result = self.dynamodb.update_file_from_table(
            "files",
            {"userId": "user1", "fileId": "file1"},
            update_expression
        )
        self.assertTrue(result)
        self.mock_table.update_item.assert_called_once()

    def test_update_file_from_table_missing_value(self):
        update_expression = {
            "expression": "SET s3Path = :val1"
            # missing 'value'
        }
        with self.assertRaises(ValueError):
            self.dynamodb.update_file_from_table(
                "files",
                {"userId": "user1", "fileId": "file1"},
                update_expression
            )

    def test_update_file_from_table_missing_expression(self):
        update_expression = {
            "value": {":val1": "new/path"}
            # missing 'expression'
        }
        with self.assertRaises(ValueError):
            self.dynamodb.update_file_from_table(
                "files",
                {"userId": "user1", "fileId": "file1"},
                update_expression
            )

    def test_update_file_from_table_client_error(self):
        self.mock_table.update_item.side_effect = ClientError(
            error_response={"Error": {"Code": "400", "Message": "Bad Request"}},
            operation_name="UpdateItem"
        )
        update_expression = {
            "expression": "SET s3Path = :val1",
            "value": {":val1": "new/path"}
        }
        result = self.dynamodb.update_file_from_table(
            "files",
            {"userId": "user1", "fileId": "file1"},
            update_expression
        )
        self.assertFalse(result)

    def test_delete_item_from_table_success(self):
        self.mock_table.delete_item.return_value = None
        result = self.dynamodb.delete_item_from_table(
            "files",
            {"userId": "user1", "fileId": "file1"}
        )
        self.assertTrue(result)
        self.mock_table.delete_item.assert_called_once()

    def test_delete_item_from_table_client_error(self):
        self.mock_table.delete_item.side_effect = ClientError(
            error_response={"Error": {"Code": "400", "Message": "Bad Request"}},
            operation_name="DeleteItem"
        )
        with self.assertRaises(ClientError):
            self.dynamodb.delete_item_from_table(
                "files",
                {"userId": "user1", "fileId": "file1"}
            )

if __name__ == "__main__":
    unittest.main()