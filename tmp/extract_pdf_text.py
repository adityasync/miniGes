import sys
from pathlib import Path

try:
    import PyPDF2
except ImportError as exc:
    sys.stderr.write("PyPDF2 is required to extract text from the PDF. Please install it in the active environment.\n")
    raise

def main(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PyPDF2.PdfReader(str(pdf_path))
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        print(f"\n--- Page {page_num} ---\n")
        print(text.strip())

if __name__ == "__main__":
    pdf_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output.pdf.pdf")
    main(pdf_arg)
