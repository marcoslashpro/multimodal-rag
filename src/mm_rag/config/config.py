from dotenv import load_dotenv
import os

config = {
  'pinecone': {
    'index_name': os.getenv("PINECONE_INDEX", 'files'),
    'namespace': os.getenv('NAMESPACE', 'mm-rag'),
    'hosting_cloud': os.getenv('HOSTING_CLOUD', 'aws'),
    'cloud_region': os.getenv("CLOUD_REGION", 'us-east-1')
  },
  'aws': {
    'bucketname': os.getenv("BUCKETNAME", 'mm-rag-bucket-may-hot')
  }
}
