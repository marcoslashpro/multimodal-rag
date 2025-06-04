import io
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from tempfile import _TemporaryFileWrapper
from mm_rag.processing.handlers import ImgHandler
from mm_rag.models.s3bucket import BucketService


class TestImgHandler(unittest.TestCase):
    def setUp(self):
        self.handler = ImgHandler()
        self.mock_bucket = MagicMock(spec=BucketService)
        self.mock_bucket.name = 'test'

    def test_adjust_orientation_no_exif(self):
        mock_image = Image.new("RGB", (100, 100))
        result = self.handler.adjust_orientation(mock_image)
        self.assertIsInstance(result, Image.Image)

    @patch("PIL.Image.Image.getexif")
    def test_adjust_orientation_with_exif(self, mock_getexif):
        mock_image = Image.new("RGB", (100, 100))
        mock_getexif.return_value = {274: 3}  # Simulate EXIF orientation
        result = self.handler.adjust_orientation(mock_image)
        self.assertIsInstance(result, Image.Image)

    def test_adjust_shape_upscale(self):
        mock_image = Image.new("RGB", (100, 100))  # Smaller than min_shape
        result = self.handler.adjust_shape(mock_image)
        self.assertEqual(result.size, (256, 256))  # Upscaled to min_shape

    def test_adjust_shape_downscale(self):
        mock_image = Image.new("RGB", (3000, 3000))  # Larger than max_shape
        result = self.handler.adjust_shape(mock_image)
        self.assertEqual(result.size, (2048, 2048))  # Downscaled to max_shape

    def test_base64_encode(self):
        mock_image = Image.new("RGB", (100, 100))
        result = self.handler.base64_encode(mock_image)
        self.assertIsInstance(result, str)

    def test_download_from_bucket_to_buffer(self):
        from io import BytesIO
        from PIL import Image

        # Create real image bytes
        img = Image.new("RGB", (10, 10), "red")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        expected_bytes = buffer.getvalue()

        # Patch the bucket method to write to a real BytesIO
        def fake_download_to_buffer(object_key, out_buffer):
            out_buffer.write(expected_bytes)
            out_buffer.seek(0)
            return out_buffer

        self.mock_bucket.download_to_buffer.side_effect = fake_download_to_buffer

        object_key = "test_image.jpg"
        img_buffer = self.handler.download_img_from_bucket_to_temp_file(object_key, self.mock_bucket)

        # Assertions
        self.mock_bucket.download_to_buffer.assert_called_once()
        self.assertEqual(img_buffer.read(), expected_bytes)
        self.assertTrue(img_buffer.readable())

    @patch("PIL.Image.open")
    def test_open_from_temp_file_valid_image(self, mock_open):
        mock_temp_file = MagicMock(spec=_TemporaryFileWrapper)
        mock_temp_file.name = "test_image.jpg"
        mock_open.return_value = Image.new("RGB", (100, 100))
        result = self.handler.open_from_temp_file(mock_temp_file)
        self.assertIsInstance(result, Image.Image)

    @patch("PIL.Image.open", side_effect=UnidentifiedImageError)
    def test_open_from_temp_file_invalid_image(self, mock_open):
        mock_temp_file = MagicMock(spec=_TemporaryFileWrapper)
        mock_temp_file.name = "test_image.jpg"
        with self.assertRaises(UnidentifiedImageError):
            self.handler.open_from_temp_file(mock_temp_file)
        mock_temp_file.close.assert_called_once()

    @patch("builtins.print")
    @patch("mm_rag.processing.handlers.ImgHandler.download_img_from_bucket")
    def test_display_valid_image(self, mock_retrieve, mock_print):
        mock_retrieve.return_value = Image.new("RGB", (100, 100))
        match_id = "test_image_id"
        self.handler.display(match_id, self.mock_bucket)
        mock_retrieve.assert_called_once_with(from_bucket=self.mock_bucket, img_key=match_id)
        mock_print.assert_called_once()

    @patch("mm_rag.processing.handlers.ImgHandler.download_img_from_bucket", side_effect=UnidentifiedImageError)
    @patch("mm_rag.processing.handlers.logger.error")
    def test_display_invalid_image(self, mock_logger, mock_retrieve):
        match_id = "invalid_image_id"
        self.handler.display(match_id, self.mock_bucket)
        mock_retrieve.assert_called_once_with(from_bucket=self.mock_bucket, img_key=match_id)
        mock_logger.assert_called_once_with(f"Trying to open a result that is not an Image: {match_id}")


if __name__ == "__main__":
    unittest.main()