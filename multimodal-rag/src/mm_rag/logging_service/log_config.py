import logging
from functools import wraps


def create_logger(name: str) -> logging.Logger:
  formatter = logging.Formatter(
    "%(asctime)s - Message level: %(levelname)s - Module: %(module)s - Line: %(lineno)d - Message: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
  )
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)

  streamHandler = logging.StreamHandler()
  streamHandler.setFormatter(formatter)
  streamHandler.setLevel(logging.DEBUG)

  logger.addHandler(streamHandler)

  return logger
