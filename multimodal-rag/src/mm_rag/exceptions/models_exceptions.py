class MissingRegionError(Exception):
  """
  This exception is particularly useful for AWS ops.
  During deployement, we will encounter some problems if we do not specify regions.
  Therefore, it is important to explain that reliably.
  """

class BucketAccessError(Exception):
  """
  Particularly useful when trying to connect to an existing bucket.
  """

class ObjectUpsertionError(Exception):
  """
  Raise this error anytime that there is a failure in the upsertion of an object into a storage.
  """

class MissingItemError(Exception):
  def __init__(self, *args: object) -> None:
    super().__init__(*args)