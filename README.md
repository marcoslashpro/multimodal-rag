# multimodal-rag (MVP)

Welcome to **multimodal-rag**, an MVP solution designed to enhance your understanding of your documents — whether they're images, text files, or PDFs — using Retrieval-Augmented Generation (RAG).

---

## 🧱 The Stack

This application was built with scalability and modularity in mind. Thanks to a class-based architecture, replacing or upgrading backend components is straightforward. Here's a breakdown of the current stack:

- **Pinecone**: Stores and retrieves vector embeddings along with metadata.
- **AWS** plays a central role in infrastructure:
  - **S3**: Stores full file objects, which are retrievable after identifying matches in Pinecone.
  - **DynamoDB**: Stores user credentials and minimal file metadata.
  - **Bedrock**: Used to run inference on the embedding model (_Titan Multimodal Embeddings G1_).
  - **Secrets Manager**: Manages sensitive variables like the Hugging Face token and Pinecone API key.
  - **CDK**: Deploys the application serverlessly to **AWS Lambda**.
- **LangChain**: Orchestrates key components like the embedder, vector store, retriever, and chatbot flow.
- **Hugging Face InferenceClient**: Powers the **VLM** (Vision-Language Model).
  - Current model: _Qwen2.5-VL-7B-Instruct_ — lightweight but capable of explaining complex queries with the right prompt.
- **FastAPI** + **Mangum**: Provides the web API layer, compatible with AWS Lambda.

---

## 🧪 Testing

Tests are split into two main categories:

- **Unit Tests**: Written using Python’s built-in `unittest` module.
- **End-to-End Tests**: Managed via `pytest`.

Robust testing ensures the application is reliable, maintainable, and easy to expand.

---

## ⚙️ Core Features

Key functionalities include:

- Upload various document types (text, images, PDFs):
  - Stored in **S3** (full file)
  - Embedded and stored in **Pinecone**
- Two main interaction modes:
  - **Search**: Retrieve relevant chunks from your knowledge base.
  - **Chat**: Interact with the VLM, which can autonomously retrieve and explain relevant information.  
    > Just ask: *“Retrieve X and explain it”* — it’ll handle the rest.

For an overview of the chat pipeline, refer to this diagram:  
📊 [mm-rag-agent-flow](https://github.com/user-attachments/assets/344b4980-ec2a-40e2-9e7b-30a970782cc8)

---

## 🚀 How to Access

As this is currently an MVP, access requires a personal token.  
If you're interested in testing or contributing, feel free to reach out:

📧 **tambascomarco35@gmail.com**

Once we chat, I may provide you with an access token and API documentation.

---

## 🧭 Roadmap

Next major milestone: **UI implementation**

Stay tuned!
