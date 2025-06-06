from pathlib import Path

from mm_rag.logging_service.log_config import create_logger


logger = create_logger(__name__)


def write_env():
  entries = {
    "PINECONE_INDEX": input("Pinecone Index Name: "),
    "PINECONE_NAMESPACE": input("Pinecone Namespace: "),
    "PINECONE_REGION": input("Pinecone Region: "),
    "AWS_BUCKET": input("S3 Bucket Name: ")
  }

  path = Path(__file__).parent.parent / '.env'

  with open(path, "w") as f:
    for key, val in entries.items():
      f.write(f"{key}={val}\n")

  logger.info(f'.env file created at path: {path}')
  print(".env file created.")

if __name__ == "__main__":
    write_env()