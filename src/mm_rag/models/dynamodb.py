import datetime
from typing import Any
import boto3
from botocore.exceptions import ClientError


class DynamoDB:
  def __init__(
      self,
  ) -> None:
    self.ddb = boto3.resource("dynamodb")

  def _validate_table(self, table_name: str):
    if not hasattr(self, table_name):
      raise ValueError(
        f"The table {table_name} does not exist and cannot be created"
      )

    return getattr(self, table_name)

  @property
  def files(self):
    try:
      # Try to create the table if it does not exists
      table = self.ddb.create_table(  # type: ignore
        TableName = 'files',
        KeySchema = [
          {
            'AttributeName': 'userId',
            'KeyType': 'HASH'
          },
          {
            'AttributeName': 'fileId',
            'KeyType': 'RANGE'
          }
        ],
        AttributeDefinitions = [
          {
            'AttributeName': 'userId',
            'AttributeType': 'S'
          },
          {
            'AttributeName': 'fileId',
            'AttributeType': 'S'
          }
        ],
        ProvisionedThroughput={
          'ReadCapacityUnits': 5,
          'WriteCapacityUnits': 5
      }
      )

      table.wait_until_exists()
      return table  # type: ignore

    # If the table already exists it raises an error
    except ClientError as e:
      # We can catch it and return the existing table for operations
      return self.ddb.Table("files")  # type: ignore

  def store_file(
    self, 
    table_name: str,
    file_id: str,
    owned_by: str,
    metadata: dict[str, Any]
    ) -> bool:

    table = self._validate_table(table_name)

    try:
      table.put_item(
        Item={
          'userId': owned_by,
          'fileId': file_id,
          'metadata': metadata
        }
      )
    except ClientError as e:
      print(e)
      return False
    return True

  def get_file_from_table(
      self,
      table_name: str,
      item_key: dict[str, str]
  ) -> dict[str, str] | None:

    table = self._validate_table(table_name)

    try:
      response = table.get_item(
        Key=item_key,
        ConsistentRead=True
        )

    except ClientError as e:
      raise ValueError(
        f"Error reading item: {e}"
      )

    if "Item" in response:
      return response['Item']

    raise ValueError(f'Key {item_key} not found in table {table_name}')

  def update_file_from_table(
      self, 
      table_name,
      item_key: dict[str, str],
      update_expression: dict[str, str | dict[str, Any]]
  ) -> bool:

    table = self._validate_table(table_name)

    attribute_values = update_expression.get(
      'value'
    )
    expression = update_expression.get(
      'expression'
    )

    if not attribute_values:
      raise ValueError(
        f"The update expression must contain a `value`"
        f"key with the value of the key to change"
        f"refer to the docs at https://boto3.amazonaws.com/v1/documentation/api/latest/guide/dynamodb.html for more info"
      )
    if not expression:
      raise ValueError(
        f"The update expression must contain a `expression`"
        f"key with the update expression"
        f"refer to the docs at https://boto3.amazonaws.com/v1/documentation/api/latest/guide/dynamodb.html for more info"
      )

    try:
      table.update_item(
        Key=item_key,
        UpdateExpression=expression,
        ExpressionAttributeValues=attribute_values
      )

    except ClientError as e:
      print(e)
      return False
    return True

  def delete_item_from_table(
      self,
      table_name: str,
      item_key: dict[str, Any]
  ) -> bool:

    table = self._validate_table(table_name)

    try:
      table.delete_item(
        Key=item_key
      )
    except ClientError as e:
      raise
    return True

  def clean(self) -> None:
    scan = self.files.scan(
      ProjectionExpression='#k, #s',
      ExpressionAttributeNames={
        '#k': 'userId',
        '#s': 'fileId'
      }
    )
    with self.files.batch_writer() as batch:
      for item in scan['Items']:
        batch.delete_item(Key={
          'userId': item['userId'],
          'fileId': item['fileId']
        })

if __name__ == '__main__':
  ddb = DynamoDB()
  # print(ddb.files.item_count)

  # Fake Metadata obj
  metadata = {
    'fileId': 'fileId1',
    # DynamoDB does not support datetime format natively
    # We'll therefor have to encode it on insertion
    # And also decode it on retrieval
    'created': datetime.datetime.now().isoformat(),
  }

  # ddb.store_file(
  #   "files",
  #   metadata,
  #   'user1'
  # )

  # print(ddb.files.item_count)
  print(ddb.get_file_from_table(
    "files",
    item_key={
      "userId": "user1",
      "fileId": "fileId1"
    }
  ))

  update_expression: dict[str, str | dict[str, Any]] = {
    'expression': "SET s3Path = :val1",
    'value': {
      ':val1': 'new/fake/path'
    }
  }

  works = ddb.update_file_from_table(
    "files",
    item_key={
      "userId": "user1",
      "fileId": "fileId1"
    },
    update_expression=update_expression
  )
  print(works)

    # print(ddb.files.item_count)
  print(ddb.get_file_from_table(
    "files",
    item_key={
      "userId": "user1",
      "fileId": "fileId1"
    }
  ))

  deleted = ddb.delete_item_from_table(
    "files",
    item_key={
      "userId": "user1",
      "fileId": "fileId1"
    }
  )
  print(deleted)
