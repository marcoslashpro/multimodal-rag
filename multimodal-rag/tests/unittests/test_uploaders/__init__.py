import mm_rag.datastructures as ds

from langchain_core.documents import Document

from PIL import Image

metadata = ds.Metadata('test_file', '.test', 'me')
mock_file = ds.File(
  metadata=metadata,
  content='test_content',
  docs=[Document(page_content='test_content', id=metadata.file_id, metadata=metadata.__dict__)],
  embeddings=[[0.1]]
)
mock_img_file = ds.File(
  metadata=metadata,
  content=Image.new('RGB', (1, 1), 'red'),
  docs=[Document(page_content='test_content', id=metadata.file_id, metadata=metadata.__dict__)],
  embeddings=[[0.1]]
)
mock_pdf_file = ds.File(
  metadata=metadata,
  content=[Image.new('RGB', (1, 1), 'red')],
  docs=[Document(page_content='test_content', id=metadata.file_id, metadata=metadata.__dict__)],
  embeddings=[[0.1]]
)