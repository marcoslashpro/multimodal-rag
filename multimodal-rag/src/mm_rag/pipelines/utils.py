import base64
import io
import os
from dataclasses import asdict
from mm_rag.exceptions import FileNotValidError, DocGenerationError
import mm_rag.pipelines.datastructures as ds

from PIL import Image
from langchain_core.documents import Document


def adjust_orientation(img: Image.Image) -> Image.Image:
    """ Adjusts image orientation based on EXIF data. """
    try:
        orientation = img.getexif().get(274)
        if orientation:

            match orientation:
                case 3:
                    img.rotate(180, expand=True)
                case 6:
                    img.rotate(270, expand=True)
                case 8:
                    img.rotate(90, expand=True)
        # Reset orientation to "Normal"

            img.getexif()[274] = 1
    except AttributeError:
        pass

    return img

def adjust_shape(img: Image.Image) -> Image.Image:
    """ Adjusts the image shape if it's too small or too large while preserving aspect ratio. """
    min_shape = 256
    max_shape = 2048

    w, h = img.size

    # First, upscale if too small (preserving aspect ratio)
    if w < min_shape or h < min_shape:

      scale = max(min_shape / w, min_shape / h)
      w, h = int(w * scale), int(h * scale)

    # Then, downscale if too large (preserving aspect ratio)
    if w > max_shape or h > max_shape:

      scale = min(max_shape / w, max_shape / h)
      w, h = int(w * scale), int(h * scale)

    return img.resize((w, h), Image.Resampling.LANCZOS)

def base64_encode(img: Image.Image) -> str:
    img_buffer = save_img_to_buffer(img)

    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")

def save_img_to_buffer(img: Image.Image) -> io.BytesIO:
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="JPEG")
    img_buffer.seek(0)

    return img_buffer

def generate_docs(ids: list[str], splits: list[str], metadata: ds.Metadata) -> list[Document]:
    if not len(ids) == len(splits):
        raise DocGenerationError(
            f"{len(ids)} ids and {len(splits)} splits must be the same."
        )
    docs: list[Document] = []

    for _id, split in zip(ids, splits):
      docs.append(
        Document(
          page_content=split,
          metadata=asdict(metadata),
          id=_id
        )
      )

    return docs

def _generate_ids(file_id: str, range_of_ids: int) -> list[str]:
    ids: list[str] = []

    for i in range(range_of_ids):
      ids.append(
        file_id + f'/chunk{i+1}'
      )

    return ids

def process_img(img: Image.Image) -> str:
      processed_img = adjust_orientation(img)
      processed_img = adjust_shape(processed_img)
      encoded_img: str = base64_encode(processed_img)

      return encoded_img
