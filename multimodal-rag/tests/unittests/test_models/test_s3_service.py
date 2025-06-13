import os
from tempfile import NamedTemporaryFile
from mm_rag.models.s3bucket import BucketService

import unittest
from unittest.mock import MagicMock, patch, call

from moto import mock_aws
import boto3

@mock_aws
class TestS3Service(unittest.TestCase):

  bucket_name = 'test-bucket'
  def setUp(self) -> None:
    self.mock_aws = mock_aws()
    self.mock_aws.start()

    client = boto3.resource('s3', region_name='eu-central-1')
    bucket = client.Bucket(self.bucket_name)
    bucket.create(CreateBucketConfiguration={'LocationConstraint': 'eu-central-1'})

    self.service = BucketService(bucket)

    self.mock_file_path = 'test.txt'
    with open(self.mock_file_path, 'w') as f:
      f.write("this is a test")

  def tearDown(self) -> None:
    self.mock_aws.stop()
    if os.path.exists(self.mock_file_path):
      os.remove(self.mock_file_path)

  def test_make_object_public_success(self):
    self.service.bucket.put_object(Key='example.txt', Body='This is a test')
    result = self.service.make_object_public("example.txt")
    self.assertTrue(result)

    acl = self.service.resource.ObjectAcl(self.bucket_name, "example.txt")
    grants = acl.grants
    self.assertTrue(
      any(
        grant['Permission'] == 'READ' and
        grant['Grantee'].get('URI') == 'http://acs.amazonaws.com/groups/global/AllUsers'
        for grant in grants
      )
    )

  def test_make_object_private_success(self):
    self.service.bucket.put_object(Key='example.txt', Body='This is a test')
    # First, make it public
    self.service.make_object_public("example.txt")
    # Then, make it private
    result = self.service.make_object_private("example.txt")
    self.assertTrue(result)

    acl = self.service.resource.ObjectAcl(self.bucket_name, "example.txt")
    grants = acl.grants
    # Ensure it's not public anymore
    self.assertFalse(
      any(
        grant['Grantee'].get('URI') == 'http://acs.amazonaws.com/groups/global/AllUsers'
        for grant in grants
      )
    )

  def test_make_object_public_nonexistent_object_raises(self):
    with self.assertRaises(ValueError) as context:
      self.service.make_object_public("nonexistent.txt")
    self.assertIn("cannot be located", str(context.exception))

  def test_make_object_private_nonexistent_object_raises(self):
    with self.assertRaises(ValueError) as context:
      self.service.make_object_private("nonexistent.txt")
    self.assertIn("cannot be located", str(context.exception))

  def test_not_upload_same_file(self):
    test_key = 'example.txt'
    test_body = 'TestBody'
    self.service.bucket.put_object(Key=test_key, Body=test_body)

    with patch.object(self.service, 'object_exists', return_value=True),\
      patch.object(self.service.bucket, 'put_object') as mock_upsert:

      self.service.upload_object_from_file(self.mock_file_path, test_key)
      with open(self.mock_file_path) as f:
        self.service.upload_object_from_file(f.read(), 'example.txt')

      mock_upsert.assert_not_called()

if __name__ == "__main__":
  unittest.main()