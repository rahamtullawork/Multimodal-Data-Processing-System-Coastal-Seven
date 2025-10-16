# utils.py
import logging
import os
import re

# -------------------------------
# Logging Setup
# -------------------------------
def setup_logger(name="multimodal_system", log_file=None, level=logging.INFO):
    """
    Setup a logger that logs to console and optional file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler (optional)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


# -------------------------------
# Configs / Constants
# -------------------------------
TEMP_DIR = os.path.join(os.getcwd(), "temp_files")
CHUNK_SIZE = 500  # default chunk size for text splitting

# Make sure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


# -------------------------------
# Helper Functions
# -------------------------------
def clean_text(text):
    """
    Clean input text by removing extra spaces, newlines, and non-printable chars.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]', '', text)  # keep printable ASCII
    return text.strip()


def split_sentences(text):
    """
    Split text into sentences using basic punctuation.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]
