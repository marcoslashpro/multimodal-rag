from mm_rag.entrypoints import upload_file, query_vectorstore, run_chatbot
from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints.setup import retriever, vlm, handler, bucket, vector_store, dynamo

import os


logger = create_logger(__name__)


def add_file() -> None:
  file_input: str = input("Insert file path: ")

  upload_file(file_input)

def query() -> None:
  query_input: str = input('Search: ')

  retrieved = query_vectorstore(query_input)

  [print(
    f"========{doc.id}========\n{doc.page_content}\n\n"
  )for doc in retrieved]

def chat() -> None:
  query = input("You: ")
  run_chatbot(query, retriever, vlm, handler, bucket)

def main() -> None:
  while True:
    choice: str = input("Search/Upload/Chat/Clean: ").lower()
    if choice == 'search':
      query()
    elif choice == 'upload':
      add_file()
    elif choice == 'chat':
      chat()
    elif choice =='clean':
      vector_store.clean()
      bucket.delete_all()
      dynamo.clean()
    else:
      print('Softly exiting...')
      quit()

if __name__ == "__main__":
  main()
