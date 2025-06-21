import asyncio

import boto3
import os
from botocore.exceptions import ClientError, ParamValidationError
from typing import Any, NewType
import io

from mm_rag.config.config import config
from mm_rag.datastructures import Storages
from mm_rag.logging_service.log_config import create_logger
from mm_rag.exceptions import MissingRegionError, BucketAccessError, ObjectUpsertionError, ObjectDeletionError


logger = create_logger(__name__)


def create_bucket(bucket_name: str, region: str = 'eu-central-1'):
  client = boto3.client('s3')
  try:
    if region is None:
      raise MissingRegionError(f'In order to create the bucket {bucket_name}, please specify also a region.')

    else:
      location = {'LocationConstraint': region}
      bucket = client.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration=location
      )
  except ClientError as e:
    if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
      return boto3.resource('s3').Bucket(bucket_name)  # type: ignore
    if e.response['Error']['Code'] == "BucketAlreadyExists":
      return boto3.resource('s3').Bucket(bucket_name)  # type: ignore
    raise e
  return bucket


class BucketService():
  def __init__(self, bucket) -> None:
    self.bucket = bucket

  @property
  def client(self):
    return boto3.client('s3')

  @property
  def resource(self):
    return boto3.resource('s3')

  @property
  def name(self) -> str:
    return config['aws'].get('bucketname', 'default-bucket-name')

  def upload_object_from_path(
      self,
      file_path: str,
      object_name=None
    ) -> bool:
    # Add multiple file upload
    if object_name is None:
      object_name = os.path.basename(file_path)

    if self.object_exists(object_name):
      return True

    try:
      self.client.upload_file(
        file_path,
        self.name,
        object_name,
      )
    except (ClientError, ParamValidationError) as e:
      raise ObjectUpsertionError(
        storage=Storages.BUCKET,
        msg=f"Error while trying to put object {file_path} in the bucket {self.bucket.name}: {str(e)}"
      ) from e

    return True

  def upload_object_from_file(
      self,
      file_obj,
      object_key: str
  ) -> bool:

    if self.object_exists(object_key):
      return True

    try:
      self.client.upload_fileobj(
        file_obj,
        self.name,
        object_key
      )
    except (ClientError, ParamValidationError) as e:
      raise ObjectUpsertionError(
        storage=Storages.BUCKET,
        msg=f"Error while trying to put object {object_key} in the bucket {self.bucket.name}: {str(e)}"
      ) from e

    return True

  def upload_object(self, key: str, body: str) -> bool:
    try:
      self.client.put_object(
        Key=key, Body=body, Bucket=self.bucket.name,
      )
    except (ClientError, ParamValidationError) as e:
      logger.error("Something is going wrong here: {}".format(e))
      raise ObjectUpsertionError(
        storage=Storages.BUCKET,
      )

    return True

  def copy_object(self, to_bucket: str, object_key: str, dest_obj_key: str | None = None) -> bool:
    copy_source = {
        'Bucket': self.name,
        'Key': object_key
    }

    if dest_obj_key is None:
      dest_obj_key = object_key

    try:
      self.client.copy(copy_source, to_bucket, dest_obj_key)
    except ClientError as e:
      logger.error(e)
      return False
    return True

  def remove_object(self, object_key: str | list[str]) -> bool:
    if isinstance(object_key, list):
      objects = [{'Key': key} for key in object_key]
    else:
      objects = [{'Key': object_key}]

    try:
      self.client.delete_objects(
        Bucket=self.bucket.name,
        Delete={
          'Objects': objects,
          'Quiet': False
        }
      )
    except Exception as e:
      logger.error(e)
      return False
    return True

  def move_object(self, to_bucket: str, object_key: str, dest_obj_key: str | None = None) -> bool:
    copied = self.copy_object(to_bucket, object_key, dest_obj_key)
    if copied:
      removed = self.remove_object(object_key)
      if removed:
        return True

    return False

  def delete(self) -> bool:
    try:
      self.client.delete_bucket(
        Bucket=self.bucket.name,
      )
    except ClientError as e:
      logger.error(e)
      return False
    return True

  def force_delete_bucket(self) -> bool:
    delete_keys: list[str] = []
    for obj in self.client.list_objects(Bucket=self.name).get('Contents'):
      delete_keys.append(obj['Key'])

    obj_removed = self.remove_object(delete_keys)
    if obj_removed:
      self.delete()
      return True
    return True

  def delete_all(self) -> bool:
    response: dict[str, Any] = self.client.list_objects(Bucket=self.bucket.name)
    contents = response.get('Contents')
    if not contents:
      raise ObjectDeletionError(Storages.BUCKET)

    obj_keys: list[dict[str, str]] = [{"Key": obj['Key']} for obj in contents]

    try:

      self.client.delete_objects(
        Bucket=self.bucket.name,
        Delete={'Objects': obj_keys}
      )
    except ClientError as e:
      logger.error(f"Error while deleting all from {self.bucket.name}: ")
      raise ObjectDeletionError(Storages.BUCKET) from e

    return True

  async def adelete_all(self):
    await asyncio.to_thread(self.delete_all)

  def make_public(self):
    """
    Renders an s3 bucket instance publicly available
    """
    try:
      self.client.put_public_access_block(
          Bucket=self.bucket.name,
          PublicAccessBlockConfiguration={
              'BlockPublicAcls': False,
              'IgnorePublicAcls': False,
              'BlockPublicPolicy': True,
              'RestrictPublicBuckets': True
          }
      )
    except ClientError as e:
      print(f'Something went wrong while making the bucket public: {str(e)}')
      return False
    return True

  def upload_public_object(self, object, obj_key: str) -> bool:
    try:
      self.upload_object_from_path(object, obj_key)
      self.make_object_public(obj_key)
    except ClientError as e:
      print(f'Something went wrong while uploading the public object {object}: {str(e)}')
      raise ObjectUpsertionError(storage=Storages.BUCKET, msg=f'Something went wrong while uploading the public object {object}: {str(e)}')
    return True
  
  def make_object_public(self, obj_key: str) -> str:
    try:
      obj_acl = self.resource.ObjectAcl(self.bucket.name, obj_key) # type: ignore
      obj_acl.put(ACL='public-read')
    except ClientError as e:
      if e.response['Error']['Code'] == 'NoSuchKey':
        raise ValueError(
          f"The provided object key: {obj_key} cannot be located in the bucket: {self.bucket.name}"
        )
      raise e  # Re-raise other ClientError
    return self.get_public_url(obj_key)

  def make_object_private(self, obj_key: str) -> bool:
    try:
      obj_acl = self.resource.ObjectAcl(self.bucket.name, obj_key) # type: ignore
      obj_acl.put(ACL='private')
    except ClientError as e:
      if e.response['Error']['Code'] == 'NoSuchKey':
        raise ValueError(
          f"The provided object key: {obj_key} cannot be located in the bucket {self.bucket.name}"
        )
      raise e  # Re-raise other ClientError
    return True

  def create_website_config(self, index_html: str) -> bool:
    """
    Args;
      index_html, (str); The path of the html file for the index page.
    """
    self.upload_public_object(index_html, 'index.html')

    try:
      self.client.put_bucket_website(
        Bucket=self.bucket.name,
        WebsiteConfiguration={
          'IndexDocument': {
            'Suffix': 'index.html'
            }
          }
        )
    except ClientError as e:
      print(f'Website configuration failed: {e}')
      return False
    return True

  def download(self, object_key: str, path: str) -> bool:
    try:
      self.client.download_file(self.bucket.name, object_key, path)
    except ClientError as e:
      raise
    return True

  def download_to_buffer(self, object_key: str, buffer: io.BytesIO) -> io.BytesIO:
    try:
      self.client.download_fileobj(self.bucket.name, object_key, buffer)
      buffer.seek(0)
    except ClientError as e:
      if e.response['Error']['Code'] == '404':
        logger.info(f"{object_key} not found in Bucket: {self.bucket.name}")
        pass
      else:
        raise e
    return buffer

  def get_public_url(self, obj_key: str) -> str:
    return f"https://{self.name}.s3.amazonaws.com/{obj_key}"

  def generate_presigned_url(self, obj_key: str, expires_in=3600) -> str:
    try:
      url = self.client.generate_presigned_url(
        "get_object",
        Params={"Bucket": self.name, "Key": obj_key},
        ExpiresIn=expires_in
      )
    except ClientError as e:
      logger.error(e)
      raise
    return url

  def object_exists(self, object_key: str) -> bool:
    try:
      self.client.get_object(
        Bucket=self.bucket.name,
        Key=object_key
      )
    except ClientError as e:
      if e.response['Error']['Code'] == 'NoSuchKey':
        return False

    logger.info(f"Not upserting {object_key} as it already exists in the bucket {self.bucket}")
    return True

if __name__ == '__main__':
  pass