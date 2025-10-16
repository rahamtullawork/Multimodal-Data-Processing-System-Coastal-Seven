# image_ocr.py
from PIL import Image
import pytesseract


def extract_text_from_image(file_path):
    """
    Extract text from image using Tesseract OCR.
    Supports PNG, JPG, JPEG.
    Returns recognized text as string.
    """
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"[ERROR] OCR failed for {file_path}: {e}")
        return ""
