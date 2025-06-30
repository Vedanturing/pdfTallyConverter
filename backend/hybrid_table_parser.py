import os
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import doctr and easyocr, with clear error messages if they are not installed
try:
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
    logger.warning("DocTR is not installed. Hybrid parser will have limited functionality. Please install it with 'pip install python-doctr[torch]'.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR is not installed. Fallback OCR functionality will not be available. Please install it with 'pip install easyocr'.")

# --- Model Initialization ---
# Initialize models only when first used to reduce startup time.
doctr_model = None
easyocr_reader = None

def get_doctr_model():
    """Initializes and returns the DocTR model."""
    global doctr_model
    if doctr_model is None and DOCTR_AVAILABLE:
        logger.info("Initializing DocTR model...")
        doctr_model = ocr_predictor(pretrained=True)
        logger.info("DocTR model initialized.")
    return doctr_model

def get_easyocr_reader():
    """Initializes and returns the EasyOCR reader."""
    global easyocr_reader
    if easyocr_reader is None and EASYOCR_AVAILABLE:
        logger.info("Initializing EasyOCR reader...")
        easyocr_reader = easyocr.Reader(['en'])  # Add other languages if needed
        logger.info("EasyOCR reader initialized.")
    return easyocr_reader

# --- Core Extraction Logic ---
def extract_tables_from_file(file_path: str) -> pd.DataFrame:
    """
    Extracts tables from a PDF or image file using a hybrid model pipeline.
    
    Args:
        file_path: Path to the PDF or image file.

    Returns:
        A pandas DataFrame containing the extracted table data.
    """
    logger.info(f"Starting table extraction for: {file_path}")
    
    if not DOCTR_AVAILABLE:
        logger.error("DocTR is not available. Cannot perform hybrid table extraction.")
        return pd.DataFrame()

    try:
        if file_path.lower().endswith('.pdf'):
            doc = DocumentFile.from_pdf(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            doc = DocumentFile.from_images(file_path)
        else:
            raise ValueError("Unsupported file type for hybrid parsing.")
        
        model = get_doctr_model()
        if not model:
            raise RuntimeError("DocTR model could not be initialized.")
            
        result = model(doc)
        
        logger.info("DocTR processing complete. Reconstructing tables.")
        
        # Reconstruct tables from DocTR output
        all_tables = reconstruct_tables_from_doctr(result)
        
        if not all_tables:
            logger.warning("DocTR did not find any tables. Trying fallback OCR.")
            return fallback_ocr(file_path)
            
        # Combine all tables into a single DataFrame
        combined_df = pd.concat(all_tables, ignore_index=True)
        
        # Normalize headers
        normalized_df = normalize_headers(combined_df)
        
        logger.info(f"Successfully extracted and normalized table with {len(normalized_df)} rows.")
        return normalized_df
        
    except Exception as e:
        logger.error(f"An error occurred during DocTR processing: {e}")
        logger.info("Falling back to EasyOCR.")
        return fallback_ocr(file_path)

def reconstruct_tables_from_doctr(result) -> list:
    """Reconstructs pandas DataFrames from DocTR's output."""
    all_tables = []
    for page in result.pages:
        for table in page.tables:
            table_data = []
            for row in table.body:
                row_data = [cell.value for cell in row.cells]
                table_data.append(row_data)
            
            # Use header if available
            if table.header:
                header_data = [cell.value for cell in table.header.cells]
                df = pd.DataFrame(table_data, columns=header_data)
            else:
                df = pd.DataFrame(table_data)
                
            all_tables.append(df)
            
    return all_tables

def fallback_ocr(file_path: str) -> pd.DataFrame:
    """Fallback OCR using EasyOCR if DocTR fails."""
    if not EASYOCR_AVAILABLE:
        logger.error("EasyOCR not available for fallback.")
        return pd.DataFrame()
        
    reader = get_easyocr_reader()
    if not reader:
        logger.error("EasyOCR reader could not be initialized.")
        return pd.DataFrame()

    logger.info("Using EasyOCR fallback.")
    try:
        if file_path.lower().endswith('.pdf'):
            # Convert PDF to images for EasyOCR
            pdf_doc = fitz.open(file_path)
            # Process first page only for fallback
            page = pdf_doc.load_page(0)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            results = reader.readtext(img_bytes)
        else:
            results = reader.readtext(file_path)
        
        # Basic reconstruction from EasyOCR results (this is non-trivial)
        # For now, return a simple DataFrame of extracted text
        if results:
            logger.info(f"EasyOCR extracted {len(results)} text blocks.")
            df = pd.DataFrame(results, columns=['bbox', 'text', 'confidence'])
            return df

    except Exception as e:
        logger.error(f"An error occurred during EasyOCR fallback: {e}")
        
    return pd.DataFrame()

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes DataFrame headers to a standard schema."""
    header_mapping = {
        'date': ['date', 'transaction date', 'txn date'],
        'description': ['description', 'narration', 'particulars'],
        'amount': ['amount', 'transaction amount', 'value'],
        'balance': ['balance', 'running balance', 'closing balance'],
        'debit': ['debit', 'withdrawal', 'dr'],
        'credit': ['credit', 'deposit', 'cr']
    }
    
    new_columns = {}
    for col in df.columns:
        col_lower = str(col).lower()
        for standard_header, variations in header_mapping.items():
            if any(var in col_lower for var in variations):
                new_columns[col] = standard_header.title()
                break
        if col not in new_columns:
            new_columns[col] = str(col).title()
            
    return df.rename(columns=new_columns) 