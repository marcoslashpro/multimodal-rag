from fastapi import FastAPI
from mangum import Mangum
import uvicorn
from mm_rag.api.routes.add_file import upload_router
from mm_rag.api.routes.search import search_router
from mm_rag.api.routes.chat import chat_router
from mm_rag.api.routes.clean import cleanup_router
from mm_rag.logging_service.log_config import create_logger
import boto3


logger = create_logger(__name__)


app = FastAPI()


app.include_router(upload_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(cleanup_router)


handler = Mangum(app)


@app.get("/")
async def root():
  return {
    "status": 200,
    "body": {
      "message": "multi-modal RAG"
    }
  }

@app.get("/debug-creds")
def debug_creds():
    session = boto3.session.Session()
    creds = session.get_credentials()
    return {
        "access_key": creds.access_key if creds else "None",
        "token": creds.token if creds else "None"
    }
