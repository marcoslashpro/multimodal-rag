import tempfile
import unittest


from mm_rag.pipelines.extractors import (
    FileNotValidError,
    generate_file_name_and_type,
    validate_path
)
import mm_rag.datastructures as ds

class DummyMetadata(ds.Metadata):
    def __init__(self):
        super().__init__(
            file_name="file",
            file_type='.txt',
            author="user1",
            created="now",
        )

class TestExtractorBase(unittest.TestCase):
    def test_validate_path_not_exists(self):
        with self.assertRaises(FileNotValidError):
            validate_path("not_a_real_file.txt")

    def test_validate_path_not_file(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotValidError):
                validate_path(d)

    def test_generate_file_name_and_type_supported(self):
        name, ftype = generate_file_name_and_type("foo.txt")
        self.assertEqual(name, "foo")
        self.assertEqual(ftype, '.txt')


if __name__ == "__main__":
    unittest.main()