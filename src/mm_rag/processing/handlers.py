import base64
import io
from sys import stdout
from typing import Any
from PIL import Image, UnidentifiedImageError

from mm_rag.logging_service.log_config import create_logger

from tempfile import NamedTemporaryFile, _TemporaryFileWrapper

from mm_rag.models.s3bucket import BucketService


logger = create_logger(__name__)


class ImgHandler:
  def adjust_orientation(self, img: Image.Image) -> Image.Image:
    """ Adjusts image orientation based on EXIF data. """
    try:
      orientation = img.getexif().get(274)
      if orientation:

        logger.debug("Orientation for image was: %d" % orientation)

        match orientation:
          case 3:
            img.rotate(180, expand=True)
          case 6:
            img.rotate(270, expand=True)
          case 8:
            img.rotate(90, expand=True)
        # Reset orientation to "Normal"

        logger.debug(f"Adjusted pic orientation")

        img.getexif()[274] = 1
    except AttributeError:
      # Handle images without EXIF data or other errors gracefully
      logger.debug(f"No orientation found for image {img}")
      pass

    return img

  def adjust_shape(self, img: Image.Image) -> Image.Image:
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

  def base64_encode(self, img: Image.Image) -> str:
    logger.debug(f"Encoding img: {img}")
    img_buffer = self.save_img_to_buffer(img)

    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")

  def save_img_to_buffer(self, img: Image.Image) -> io.BytesIO:
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="JPEG")
    img_buffer.seek(0)

    return img_buffer

  def download_img_from_bucket(self, img_key: str, from_bucket: BucketService) -> Image.Image:
    logger.debug(f"Downloaded object {img_key} from {from_bucket.name}")
    img_buffer = self.download_img_from_bucket_to_temp_file(img_key, from_bucket)
    return self.open_img_from_buffer(img_buffer)

  def open_img_from_buffer(self, buffer: io.BytesIO) -> Image.Image:
      return Image.open(buffer)


  def download_img_from_bucket_to_temp_file(self,  object_key: str, bucket: BucketService) -> io.BytesIO:
    img_buffer = bucket.download_to_buffer(object_key, io.BytesIO())

    return img_buffer

  def open_from_temp_file(self, temp_file: _TemporaryFileWrapper) -> Image.Image:
    try:
      img = Image.open(temp_file.name)
    except UnidentifiedImageError as e:
      temp_file.close()
      raise
    temp_file.close()

    return img

  def display(self, match_id: str, bucket: BucketService) -> None:
    try:
      logger.debug(f'Retrieving {match_id} from bucket')
      img: Image.Image = self.download_img_from_bucket(from_bucket=bucket, img_key=match_id)
      print(f'\n\n======Image{match_id}=======\n{img.show()}')

    except UnidentifiedImageError as e:
      logger.error(f"Trying to open a result that is not an Image: {match_id}")
      pass
