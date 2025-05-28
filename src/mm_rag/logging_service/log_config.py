import logging
from pathlib import Path


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

  # Get the absolute path to the logging_service directory
  logs_dir = Path(__file__).parent / "logs"
  logs_dir.mkdir(exist_ok=True)  # Make sure the 'logs' directory exists

  log_file_path = logs_dir / f"{name}.log"

  fileHandler = logging.FileHandler(log_file_path, encoding="utf-8")
  fileHandler.setFormatter(formatter)
  fileHandler.setLevel(logging.DEBUG)

  logger.addHandler(streamHandler)
  logger.addHandler(fileHandler)

  return logger