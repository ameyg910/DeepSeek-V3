import pytest

from data.pdf_extraction import (clean_text, extract_pdf_text,)

def test_clean_whitespace():
    text = "Hello   world.\n\n\nNext paragraph."

    cleaned = clean_text(text)

    assert cleaned == "Hello world.\n\nNext paragraph."

def test_join_hyphenated_words():
    text = "differen-\ntiation"
    cleaned = clean_text(text)

    assert cleaned == "differentiation"

def test_preserve_math_notation():
    text = "f(x) = x^2 - 3x + 2"

    cleaned = clean_text(text)

    assert cleaned == text

def test_missing_pdf():
    with pytest.raises(FileNotFoundError):
        extract_pdf_text("missing_book.pdf")