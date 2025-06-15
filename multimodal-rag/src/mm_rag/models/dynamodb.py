import datetime
from typing import Any
import boto3
import boto3.dynamodb
import boto3.dynamodb.conditions
from botocore.exceptions import ClientError

from mm_rag.exceptions import MissingItemError
from mm_rag.logging_service.log_config import create_logger


logger = create_logger(__name__)


class DynamoDB:
  def __init__(
      self,
  ) -> None:
    self.ddb = boto3.resource("dynamodb", region_name='eu-central-1')

  def _validate_table(self, table_name: str):
    if not hasattr(self, table_name):
      raise ValueError(
        f"The table {table_name} does not exist and cannot be created"
      )

    return getattr(self, table_name)

  @property
  def users(self):
    try:
      # Try to create the table if it does not exists
      table = self.ddb.create_table(  # type: ignore
        TableName = 'users',
        KeySchema = [
          {
            'AttributeName': 'userId',
            'KeyType': 'HASH'
          },
          {
            'AttributeName': 'PAT',
            'KeyType': 'RANGE'
          }
        ],
        AttributeDefinitions = [
          {
            'AttributeName': 'userId',
            'AttributeType': 'S'
          },
          {
            'AttributeName': 'PAT',
            'AttributeType': 'S'
          },
          {
            "AttributeName": "PAT-gsi",
            "AttributeType": "S"
          }
        ],
        GlobalSecondaryIndexes=[
          {
            "IndexName": "PAT-index",
            "KeySchema": [
              {
                "AttributeName": "PAT-gsi",
                "KeyType": "HASH"
              }
            ],
            "Projection": {
              "ProjectionType": "ALL",
            },
            "ProvisionedThroughput": {
              'ReadCapacityUnits': 5,
              'WriteCapacityUnits': 5
            }
          }
        ],
        ProvisionedThroughput={
          'ReadCapacityUnits': 5,
          'WriteCapacityUnits': 5
      }
      )

      table.wait_until_exists()
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            # Already exists
            table = self.ddb.Table('users')  # type: ignore
        else:
            raise
    return table

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

  def add_user(
      self,
      table_name: str,
      PAT: str,
      user_id: str
  ) -> bool:

    table = self._validate_table(table_name)

    try:
      table.put_item(
        Item={
          "userId": user_id,
          "PAT": PAT,
          "PAT-gsi": PAT
        }
      )
    except ClientError as e:
      raise e
    return True

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

  def get_from_table(
      self,
      table_name: str,
      item_key: dict[str, str]
  ) -> dict[str, str]:

    table = self._validate_table(table_name)

    try:
      response = table.get_item(
        Key=item_key,
        ConsistentRead=True
        )
    except ClientError as e:
      if e.response['Error']['Code'] == "ResourceNotFoundException":
        logger.error(f"Item {item_key} not found in table {table_name}")
        raise MissingItemError(f'Key {item_key} not found in table {table_name}')

      logger.error(f"Error while retrieving {item_key} from {table_name}: {str(e)}")
      logger.error("i do not know why this function is being called")
      raise e 

    if "Item" in response:
      return response['Item']

    raise MissingItemError(f'Key {item_key} not found in table {table_name}')

  def query_with_gsi(self, table_name: str, index_name: str, gsi_key: str, gsi_value: str):
    table = self._validate_table(table_name)

    try:
      logger.debug(f"Quering {table_name}, index: {index_name}, with {gsi_key} = {gsi_value}")
      response = table.query(
        IndexName=index_name,
        KeyConditionExpression=boto3.dynamodb.conditions.Key(gsi_key).eq(gsi_value)
      )
    except ClientError as e:
      if e.response['Error']['Code'] == 'ResourceNotFoundException':
        logger.error(f"Error while querying the table: {table_name}, at index: {index_name}, with {gsi_key}: {gsi_value}")
        raise MissingItemError(
          f"Item {gsi_value} not found in index {index_name}, "
        )

      logger.error(f'Error while querying Dynamo with args: IndexName: {index_name}\nGsi Key: {gsi_key}, Value: {gsi_value}. Error: {str(e)}')
      raise e

    if not 'Items' in response:
      raise MissingItemError(f"no items in response: {response}")

    items = response['Items']

    if len(items) < 1:
      raise MissingItemError(f"Item {gsi_value} not found in index {index_name}, ")

    return items

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
  print(ddb.get_from_table(
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
  print(ddb.get_from_table(
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
