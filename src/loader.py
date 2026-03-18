from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file.
    """
    reader = PdfReader(pdf_path)
    text_data = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_data.append(text)

    return "\n".join(text_data)
