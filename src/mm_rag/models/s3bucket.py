import boto3
import os
from botocore.exceptions import ClientError
from typing import Any, NewType
import io

from mm_rag.config.config import config
from mm_rag.logging_service.log_config import create_logger

BucketName = NewType('BucketName', str)


logger = create_logger(__name__)


class Bucket():
  def __init__(self, region: str = 'eu-central-1') -> None:
    self.client = self.create_client('s3')
    self.name = config['aws'].get('bucketname', 'default-bucket-name')
    self.region = region
    self.resource = boto3.resource('s3')

    # Fix the bucket creation logic
    try:
      self.bucket = self.create_bucket(self.name, self.region)
    except ClientError as e:
      if 'BucketAlreadyOwnedByYou' in str(e):
        self.bucket = boto3.resource('s3').Bucket(self.name)  # type: ignore

  def create_bucket(self, bucket_name: BucketName, region: str) -> bool:
    try:
      if region is None:
        bucket = self.client.create_bucket(Bucket=bucket_name)
      else:
        location = {'LocationConstraint': region}
        bucket = self.client.create_bucket(
          Bucket=bucket_name,
          CreateBucketConfiguration=location
        )
    except ClientError as e:
      raise
    return bucket

  def create_client(self, service: str):
    return boto3.client(service)


class BucketService():
  def __init__(self, bucket: Bucket) -> None:
    self.bucket = bucket

  @property
  def name(self) -> str:
    return self.bucket.name

  def upload_object_from_path(
      self,
      file_path: str,
      object_name=None
    ) -> bool:
    # Add multiple file upload
    if object_name is None:
      object_name = os.path.basename(file_path)

    try:
      self.bucket.client.upload_file(
        file_path,
        self.bucket.name,
        object_name,
        # Use 'ExtraArgs for metadata: : dict[str, dict[str, Any]]
        # ExtraArgs = {
          # 'Metadata': {
            # 'mykey': 'myvalue'
          # }
        # }
        )
    except ClientError as e:
      logger.error(e)
      return False
    return True

  def upload_object_from_file(
      self,
      file_obj,
      object_key: str
  ) -> bool:

    try:
      self.bucket.client.upload_fileobj(
        file_obj,
        self.bucket.name,
        object_key
      )
    except ClientError as e:
      raise
    return True


  def copy_object(self, to_bucket: BucketName, object_key: str, dest_obj_key: str | None = None) -> bool:
    copy_source = {
        'Bucket': self.bucket.name,
        'Key': object_key
    }

    if dest_obj_key is None:
      dest_obj_key = object_key

    try:
      self.bucket.client.copy(copy_source, to_bucket, dest_obj_key)
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
      self.bucket.client.delete_objects(
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

  def move_object(self, to_bucket: BucketName, object_key: str, dest_obj_key: str | None = None) -> bool:
    copied = self.copy_object(to_bucket, object_key, dest_obj_key)
    if copied:
      removed = self.remove_object(object_key)
      if removed:
        return True

    return False

  def delete(self) -> bool:
    try:
      self.bucket.client.delete_bucket(
        Bucket=self.bucket.name,
      )
    except ClientError as e:
      logger.error(e)
      return False
    return True

  def force_delete_bucket(self) -> bool:
    delete_keys: list[str] = []
    for obj in self.bucket.client.list_objects(Bucket=self.bucket.name).get('Contents'):
      delete_keys.append(obj['Key'])

    obj_removed = self.remove_object(delete_keys)
    if obj_removed:
      self.delete()
      return True
    return True

  def delete_all(self) -> bool:
    try:
      response: dict[str, Any] = self.bucket.client.list_objects(Bucket=self.bucket.name)
      obj_keys: list[dict[str, str]] = [{"Key": obj['Key']} for obj in response['Contents']]

      self.bucket.client.delete_objects(
        Bucket=self.bucket.name,
        Delete={'Objects': obj_keys}
      )
    except ClientError as e:
      logger.error(f"Error while deleting all from {self.bucket.name}: ")
      raise e
    return True

  def make_public(self):
    """
    Renders an s3 bucket instance publicly available
    """
    try:
      self.bucket.client.put_public_access_block(
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
      obj_acl = self.bucket.resource.ObjectAcl(self.bucket.name, 'index.html') # type: ignore
      obj_acl.put(ACL='public-read')
    except ClientError as e:
      print(f'Something went wrong while uploading the public object {object}: {str(e)}')
      return False
    return True

  def create_website_config(self, index_html: str) -> bool:
    """
    Args;
      index_html, (str); The path of the html file for the index page.
    """
    self.upload_public_object(index_html, 'index.html')

    try:
      self.bucket.client.put_bucket_website(
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
      self.bucket.client.download_file(self.bucket.name, object_key, path)
    except ClientError as e:
      raise
    return True

  def download_to_buffer(self, object_key: str, buffer: io.BytesIO) -> io.BytesIO:
    try:
      self.bucket.client.download_fileobj(self.bucket.name, object_key, buffer)
      buffer.seek(0)
    except ClientError as e:
      if e.response['Error']['Code'] == '404':
        logger.info(f"{object_key} not found in Bucket: {self.bucket.name}")
        pass
      else:
        raise e
    return buffer

if __name__ == '__main__':
  bucket = Bucket()
  print(f'Bucket {bucket.name} Connected.')
  service = BucketService(bucket=bucket)

  service.remove_object('index.html')

  service.create_website_config('/home/marco/Projects/mm-rag/src/mm_rag/server/templates/index.html')