from fastapi import FastAPI
from mangum import Mangum
import uvicorn
from mm_rag.app.routes.views.add_file import upload_router
from mm_rag.app.routes.views.search import search_router
from mm_rag.app.routes.views.chat import chat_router
from mm_rag.app.routes.views.clean import cleanup_router
from mm_rag.logging_service.log_config import create_logger


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


if __name__ == "__main__":
  uvicorn.run(app)