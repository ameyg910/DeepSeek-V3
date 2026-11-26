import re 
from pathlib import Path 

import fitz 

def extract_pdf_text(pdf_path: str | Path) -> str: 
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf": 
        raise ValueError("Input file must be a PDF")

    pages = []

    with fitz.open(pdf_path) as document: 
        for page in document: 
            page_text = page.get_text("text", sort=True)
            pages.append(page_text)
    return "\n\n".join(pages)

def clean_text(text: str) -> str: 
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = text.replace("\f", "\n")
    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
        "",
        text,
    )
    lines = [
        line.strip()
        for line in text.split("\n")
    ]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def extract_and_clean_pdf(pdf_path: str | Path,) -> str: 
    raw_text = extract_pdf_text(pdf_path)
    return clean_text(raw_text)

def save_text(text: str, output_path: str | Path,):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True,)
    output_path.write_text(text, encoding="utf-8",)

if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ").strip()

    text = extract_and_clean_pdf(pdf_path)

    output_path = Path("data/textbook.txt")

    save_text(text, output_path)

    print(f"Extracted {len(text)} characters")
    print(f"Saved text to {output_path}")