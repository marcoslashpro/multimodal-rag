import pytest
from unittest.mock import MagicMock, patch

from mm_rag.pipelines.extractors import CodeExtractor
from mm_rag.exceptions import FileNotValidError

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


splitter = RecursiveCharacterTextSplitter()
extractor = CodeExtractor()


# Map Code enum extensions to Language enum for langchain_text_splitters
CODE_LANG_MAP = {
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.cpp': Language.CPP,
    '.cs': Language.CSHARP,
    '.go': Language.GO,
    '.html': Language.HTML,
    '.java': Language.JAVA,
    '.kt': Language.KOTLIN,
    '.lua': Language.LUA,
    '.md': Language.MARKDOWN,
    '.php': Language.PHP,
    '.rb': Language.RUBY,
    '.rs': Language.RUST,
    '.scala': Language.SCALA,
    '.swift': Language.SWIFT,
    '.tex': Language.LATEX,
    '.ts': Language.TS,
}

@pytest.mark.parametrize('ext, exp_splitter', [
    (ext, splitter.from_language(lang))
    for ext, lang in CODE_LANG_MAP.items()
])
def test_create_splitter_success(ext, exp_splitter):
    assert extractor._create_splitter(ext)._separators == exp_splitter._separators


@pytest.mark.parametrize('ext', [
    '.unknown', '', '.foo', '.bar'
])
def test_create_splitter_invalid(ext):
    with pytest.raises(FileNotValidError):
        extractor._create_splitter(ext)