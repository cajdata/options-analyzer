# app/ocr.py

import pytesseract
import cv2
import numpy as np
import tempfile

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Given raw image bytes, convert to OpenCV image, preprocess, then OCR.
    Returns raw text from the screenshot.
    """
    # Write bytes to a temporary file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_bytes)
        temp_path = tmp.name

    # Load with OpenCV
    img = cv2.imread(temp_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image bytes.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Simple threshold to remove background noise
    gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # OCR config
    config = "--psm 6"  # 6 = Assume a single uniform block of text
    text = pytesseract.image_to_string(gray, config=config)

    return text
