from mm_rag.entrypoints import upload_file, query_vectorstore, run_chatbot, cleanup
from mm_rag.logging_service.log_config import create_logger
from mm_rag.entrypoints.setup import vlm, handler, bucket, retriever_factory, vector_store_factory


logger = create_logger(__name__)

def add_file(user_id: str) -> None:
  file_input: str = input("Insert file path: ")

  upload_file(file_input, namespace=user_id)

def query(user_id: str) -> None:
  query_input: str = input('Search: ')

  retrieved = query_vectorstore(query_input, namespace=user_id)

  [print(
    f"========{doc.id}========\n{doc.page_content}\n\n"
  )for doc in retrieved]

def chat(user_id: str) -> None:
  retriever = retriever_factory.get_retriever(vector_store=vector_store_factory.get_vector_store(namespace=user_id))
  query = input("You: ")
  run_chatbot(query, retriever, vlm, handler, bucket)

def main() -> None:
  user_id: str = input("Enter your userId:")
  while True:
    choice: str = input("Search/Upload/Chat/Clean: ").lower()
    if choice == 'search':
      query(user_id=user_id)
    elif choice == 'upload':
      add_file(user_id=user_id)
    elif choice == 'chat':
      chat(user_id=user_id)
    elif choice =='clean':
      cleanup(namespace=user_id)
    else:
      print('Softly exiting...')
      quit()

if __name__ == "__main__":
  main()