import tempfile
import unittest


from mm_rag.pipelines.extractors import (
    FileNotValidError
)
import mm_rag.pipelines.datastructures as ds

class DummyMetadata(ds.Metadata):
    def __init__(self):
        super().__init__(
            file_name="file",
            file_type=ds.FileType(".txt").value,
            author="user1",
            created="now",
        )

class TestExtractorBase(unittest.TestCase):
    def test_validate_path_not_exists(self):
        from mm_rag.pipelines.extractors import Extractor
        with self.assertRaises(FileNotValidError):
            Extractor._validate_path("not_a_real_file.txt")

    def test_validate_path_not_file(self):
        from mm_rag.pipelines.extractors import Extractor
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotValidError):
                Extractor._validate_path(d)

    def test_generate_file_name_and_type_supported(self):
        from mm_rag.pipelines.extractors import Extractor
        name, ftype = Extractor._generate_file_name_and_type("foo.txt")
        self.assertEqual(name, "foo")
        self.assertIsInstance(ftype, ds.FileType)

    def test_generate_file_name_and_type_unsupported(self):
        from mm_rag.pipelines.extractors import Extractor
        with self.assertRaises(FileNotValidError):
            Extractor._generate_file_name_and_type("foo.unsupported")


if __name__ == "__main__":
    unittest.main()