import pdfplumber
import pymupdf as fitz  # PyMuPDF
import pytesseract
import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
import traceback
import cv2
import io
from PyPDF2 import PdfReader
import tempfile
import os
import easyocr
from table_detector import TableDetector

logger = logging.getLogger(__name__)

@dataclass
class BankStatementConfig:
    """Configuration for different bank statement formats"""
    bank_name: str
    date_formats: List[str]
    amount_patterns: List[str]
    header_variations: Dict[str, List[str]]
    table_keywords: List[str]

# Common Indian bank configurations
BANK_CONFIGS = {
    "ICICI": BankStatementConfig(
        bank_name="ICICI",
        date_formats=["%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"],
        amount_patterns=[
            r"[₹\$]?\s*([0-9,]+\.?\d*)",
            r"([0-9,]+\.?\d*)\s*Dr",
            r"([0-9,]+\.?\d*)\s*Cr"
        ],
        header_variations={
            "date": ["Date", "Tran Date", "Transaction Date", "Value Date"],
            "description": ["Description", "Narration", "Particulars", "Details"],
            "debit": ["Debit", "Withdrawal (Dr)", "DR", "Withdrawal Amount"],
            "credit": ["Credit", "Deposit (Cr)", "CR", "Deposit Amount"],
            "balance": ["Balance", "Running Balance", "Closing Balance"]
        },
        table_keywords=["Transaction Date", "Value Date", "Balance"]
    ),
    "SBI": BankStatementConfig(
        bank_name="SBI",
        date_formats=["%d %b %Y", "%d-%m-%Y", "%d/%m/%Y"],
        amount_patterns=[
            r"[₹\$]?\s*([0-9,]+\.?\d*)",
            r"([0-9,]+\.?\d*)\s*Dr",
            r"([0-9,]+\.?\d*)\s*Cr"
        ],
        header_variations={
            "date": ["Txn Date", "Value Date", "Date", "Transaction Date"],
            "description": ["Description", "Particulars", "Narration", "Details"],
            "debit": ["Debit", "Withdrawal", "DR"],
            "credit": ["Credit", "Deposit", "CR"],
            "balance": ["Balance", "Running Balance"]
        },
        table_keywords=["Txn Date", "Description", "Balance"]
    ),
    # Add more banks as needed
}

def clean_amount(amount_str: str) -> float:
    """Convert amount string to float, handling various formats"""
    if not amount_str or pd.isna(amount_str):
        return 0.0
    
    # Convert to string if not already
    amount_str = str(amount_str).strip()
    
    # Remove currency symbols, spaces
    cleaned = re.sub(r'[₹\$\s]', '', amount_str)
    
    # Handle negative indicators
    is_negative = False
    if '(' in cleaned and ')' in cleaned:  # (100.00) format
        is_negative = True
        cleaned = cleaned.replace('(', '').replace(')', '')
    elif cleaned.startswith('-'):
        is_negative = True
        cleaned = cleaned[1:]
    elif cleaned.lower().endswith('dr'):
        is_negative = True
        cleaned = cleaned[:-2]
    elif cleaned.lower().endswith('cr'):
        cleaned = cleaned[:-2]
    
    # Remove commas and any other non-numeric chars except decimal point
    cleaned = re.sub(r'[^0-9.]', '', cleaned)
    
    try:
        amount = float(cleaned)
        return -amount if is_negative else amount
    except ValueError:
        logger.warning(f"Could not convert amount: {amount_str}")
        return 0.0

def normalize_headers(headers: List[str], bank_config: BankStatementConfig) -> Dict[str, str]:
    """Map variant headers to standard names"""
    normalized = {}
    for std_name, variations in bank_config.header_variations.items():
        for header in headers:
            if any(var.lower() in header.lower() for var in variations):
                normalized[header] = std_name
                break
    return normalized

def detect_bank_type(text: str) -> Optional[str]:
    """Detect bank type from statement content"""
    bank_identifiers = {
        "ICICI": ["ICICI Bank", "ICICI Bank Statement", "www.icicibank.com"],
        "SBI": ["State Bank of India", "SBI", "www.onlinesbi.com"],
        # Add more banks as needed
    }
    
    for bank, identifiers in bank_identifiers.items():
        if any(identifier.lower() in text.lower() for identifier in identifiers):
            return bank
    return None

def process_bank_statement(file_path: str, password: Optional[str] = None) -> pd.DataFrame:
    """Process bank statement and return structured data"""
    logger.info(f"Processing bank statement: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()

    try:
        # First try to validate and decrypt PDF if needed
        try:
            with fitz.open(file_path) as doc:
                if doc.needs_pass:
                    if not password:
                        logger.error("PDF is password protected but no password provided")
                        return pd.DataFrame()
                    if not doc.authenticate(password):
                        logger.error("Invalid password for PDF")
                        return pd.DataFrame()
                if doc.page_count == 0:
                    logger.error("PDF has no pages")
                    return pd.DataFrame()
        except Exception as e:
            logger.warning(f"PyMuPDF validation failed: {str(e)}, trying PyPDF2")
            try:
                with open(file_path, 'rb') as f:
                    reader = PdfReader(f)
                    if reader.is_encrypted:
                        if not password:
                            logger.error("PDF is password protected but no password provided")
                            return pd.DataFrame()
                        try:
                            reader.decrypt(password)
                        except:
                            logger.error("Invalid password for PDF")
                            return pd.DataFrame()
                    if len(reader.pages) == 0:
                        logger.error("PDF has no pages")
                        return pd.DataFrame()
            except Exception as e:
                logger.error(f"PDF validation failed: {str(e)}")
                return pd.DataFrame()

        # Try pdfplumber first
        try:
            logger.info("Attempting to parse with pdfplumber...")
            with pdfplumber.open(file_path, password=password) as pdf:
                all_tables = []
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num} of {len(pdf.pages)}")
                    
                    # Try different table extraction settings
                    settings_list = [
                        {},  # Default settings
                        {"vertical_strategy": "text", "horizontal_strategy": "text"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "text_tolerance": 3,
                            "text_x_tolerance": 3,
                            "intersection_tolerance": 3,
                        }
                    ]
                    
                    tables = []
                    for settings in settings_list:
                        if not tables:  # Only try next settings if no tables found
                            try:
                                if settings:
                                    tables = page.find_tables(settings).extract()
                                else:
                                    tables = page.extract_tables()
                                if tables:
                                    logger.info(f"Found {len(tables)} tables with settings: {settings}")
                            except Exception as e:
                                logger.warning(f"Table extraction failed with settings {settings}: {str(e)}")
                    
                    if tables:
                        all_tables.extend(tables)
                    else:
                        logger.warning(f"No tables found on page {page_num}")
                
                if all_tables:
                    # Process tables into DataFrame
                    headers = all_tables[0][0]
                    rows = []
                    for table in all_tables:
                        rows.extend(table[1:])  # Skip header row for subsequent tables
                    
                    df = pd.DataFrame(rows, columns=headers)
                    
                    # Clean up the data
                    df = df.replace(r'^\s*$', '', regex=True)  # Replace whitespace-only cells
                    df = df.replace(r'\s+', ' ', regex=True)   # Normalize whitespace
                    df = df.fillna('')  # Replace NaN with empty string
                    
                    # Clean up column names
                    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
                    
                    # Try to identify and clean up common columns
                    for col in df.columns:
                        if any(keyword in col for keyword in ['amount', 'debit', 'credit', 'balance']):
                            df[col] = df[col].apply(clean_amount)
                        elif any(keyword in col for keyword in ['date']):
                            df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')
                            df[col] = df[col].dt.strftime('%Y-%m-%d')
                    
                    if len(df.columns) >= 3 and len(df) > 0:
                        logger.info("Successfully extracted table data")
                        return df
                    else:
                        logger.warning("Extracted table data appears invalid")
        
        except Exception as e:
            logger.error(f"pdfplumber parsing failed: {str(e)}")
        
        # If pdfplumber fails, try bank-specific parsing
        try:
            logger.info("Attempting bank-specific parsing...")
            with fitz.open(file_path) as doc:
                if doc.needs_pass:
                    doc.authenticate(password)
                
                text_content = []
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text_content.append(page.get_text())
                
                full_text = '\n'.join(text_content)
                bank_type = detect_bank_type(full_text)
                
                if bank_type and bank_type in BANK_CONFIGS:
                    logger.info(f"Detected bank type: {bank_type}")
                    config = BANK_CONFIGS[bank_type]
                    
                    data = []
                    for text in text_content:
                        lines = text.split('\n')
                        for line in lines:
                            row = {}
                            
                            # Try to extract date
                            for date_format in config.date_formats:
                                try:
                                    date_match = re.search(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}', line)
                                    if date_match:
                                        date_str = date_match.group(0)
                                        row['date'] = datetime.strptime(date_str, date_format).strftime('%Y-%m-%d')
                                        break
                                except ValueError:
                                    continue
                            
                            # Try to extract amounts
                            for pattern in config.amount_patterns:
                                amount_match = re.search(pattern, line)
                                if amount_match:
                                    amount = clean_amount(amount_match.group(1))
                                    if 'amount' not in row:
                                        row['amount'] = amount
                            
                            # Extract description
                            if row:
                                description = re.sub(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}', '', line)
                                description = re.sub(r'[₹\$]?\s*[0-9,]+\.?\d*\s*(Dr|Cr)?', '', description).strip()
                                row['description'] = description
                                data.append(row)
                    
                    if data:
                        df = pd.DataFrame(data)
                        logger.info("Successfully extracted data using bank-specific parsing")
                        return df
                    else:
                        logger.warning("No data found using bank-specific parsing")
        
        except Exception as e:
            logger.error(f"Bank-specific parsing failed: {str(e)}")
        
        # If both methods fail, try OCR as last resort
        try:
            logger.info("Attempting OCR processing...")
            with fitz.open(file_path) as doc:
                if doc.needs_pass:
                    doc.authenticate(password)
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Convert to grayscale and improve contrast
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    
                    # Convert back to PIL Image
                    img = Image.fromarray(img)
                    
                    try:
                        text = pytesseract.image_to_string(img)
                        if text.strip():
                            # Process OCR text similar to bank-specific parsing
                            bank_type = detect_bank_type(text)
                            if bank_type and bank_type in BANK_CONFIGS:
                                config = BANK_CONFIGS[bank_type]
                                # Extract data using same logic as bank-specific parsing
                                # ... (same data extraction code)
                                pass
                    except Exception as e:
                        logger.error(f"OCR failed for page {page_num}: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
        
        logger.warning("All parsing methods failed")
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Bank statement processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

class PDFPasswordError(Exception):
    """Custom exception for password-protected PDFs"""
    pass

class BankStatementParser:
    def __init__(self):
        self.missed_rows = []
        self.missed_metadata_lines = []
        self.table_detector = TableDetector()
        self.bank_patterns = {
            'sbi': {
                'identifier': r'State Bank of India',
                'date_pattern': r'\d{2}/\d{2}/\d{4}',
                'amount_pattern': r'(?:(?:Rs|INR|₹)\s*)?[\d,]+\.\d{2}',
                'balance_pattern': r'(?:Balance|Bal)\.?\s*(?:Rs|INR|₹)?\s*[\d,]+\.\d{2}'
            },
            'hdfc': {
                'identifier': r'HDFC Bank',
                'date_pattern': r'\d{2}/\d{2}/\d{4}',
                'amount_pattern': r'(?:(?:Rs|INR|₹)\s*)?[\d,]+\.\d{2}',
                'balance_pattern': r'(?:Balance|Bal)\.?\s*(?:Rs|INR|₹)?\s*[\d,]+\.\d{2}'
            },
            'icici': {
                'identifier': r'ICICI Bank',
                'date_pattern': r'\d{2}/\d{2}/\d{4}',
                'amount_pattern': r'(?:(?:Rs|INR|₹)\s*)?[\d,]+\.\d{2}',
                'balance_pattern': r'(?:Balance|Bal)\.?\s*(?:Rs|INR|₹)?\s*[\d,]+\.\d{2}'
            },
            'axis': {
                'identifier': r'Axis Bank',
                'date_pattern': r'\d{2}/\d{2}/\d{4}',
                'amount_pattern': r'(?:(?:Rs|INR|₹)\s*)?[\d,]+\.\d{2}',
                'balance_pattern': r'(?:Balance|Bal)\.?\s*(?:Rs|INR|₹)?\s*[\d,]+\.\d{2}'
            }
        }

    def detect_bank(self, text: str) -> Optional[str]:
        """Detect bank based on text patterns"""
        for bank, patterns in self.bank_patterns.items():
            if re.search(patterns['identifier'], text, re.IGNORECASE):
                return bank
        return None

    def extract_tables_from_pdf(self, pdf_path: str, password: Optional[str] = None) -> List[pd.DataFrame]:
        """Extract tables from PDF using multiple methods"""
        tables = []
        try:
            # Try PyMuPDF first for image extraction
            doc = fitz.open(pdf_path)
            if password:
                doc.authenticate(password)

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_np = np.array(img)
                
                # Process image with table detector
                df, discarded = self.table_detector.process_image(img_np)
                if df is not None and not df.empty:
                    tables.append(df)
                    self.missed_metadata_lines.extend(discarded)

            # If no tables found, try pdfplumber
            if not tables:
                with pdfplumber.open(pdf_path, password=password) as pdf:
                    for page in pdf.pages:
                        extracted_tables = page.extract_tables()
                        for table in extracted_tables:
                            if table:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                tables.append(df)

        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []

        return tables

    def process_bank_statement(self, pdf_path: str, password: Optional[str] = None) -> Dict[str, Any]:
        """Process bank statement and extract transaction data"""
        try:
            # Extract text for bank detection
            doc = fitz.open(pdf_path)
            if password:
                doc.authenticate(password)
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Detect bank
            bank_type = self.detect_bank(text)
            
            # Extract tables
            tables = self.extract_tables_from_pdf(pdf_path, password)
            
            if not tables:
                return {
                    'success': False,
                    'error': 'No tables found in the document',
                    'missed_metadata_lines': self.missed_metadata_lines
                }
            
            # Combine all tables
            df = pd.concat(tables, ignore_index=True)
            
            # Clean and standardize data
            df = self.clean_data(df)
            
            return {
                'success': True,
                'data': df.to_dict('records'),
                'bank_type': bank_type,
                'missed_metadata_lines': self.missed_metadata_lines
            }
            
        except Exception as e:
            logger.error(f"Error processing bank statement: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'missed_metadata_lines': self.missed_metadata_lines
            }

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the extracted data"""
        if df.empty:
            return df
        
        # Remove any unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Convert amount columns to numeric
        for col in ['debit', 'credit', 'balance']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x).replace(',', '').replace('₹', '').strip())
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date')
        
        return df

def validate_numeric_fields(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Validate numeric fields in the DataFrame"""
    errors = []
    warnings = []
    
    # Define numeric columns to validate
    numeric_columns = ['amount', 'balance', 'debit', 'credit']
    
    for col in df.columns:
        if any(num_col in col.lower() for num_col in numeric_columns):
            for idx, value in df[col].items():
                if pd.isna(value) or value == '':
                    continue
                    
                try:
                    # Try to convert to float
                    float_val = float(str(value).replace(',', '').replace('₹', '').strip())
                    
                    # Check for reasonable ranges
                    if float_val == 0:
                        warnings.append({
                            'row': idx,
                            'column': col,
                            'value': value,
                            'message': f"Zero value found in {col}"
                        })
                    elif abs(float_val) > 10000000:  # Value > 1 crore
                        warnings.append({
                            'row': idx,
                            'column': col,
                            'value': value,
                            'message': f"Very large value ({float_val}) found in {col}"
                        })
                        
                except ValueError:
                    errors.append({
                        'row': idx,
                        'column': col,
                        'value': value,
                        'message': f"Invalid numeric value in {col}"
                    })
    
    return {
        'errors': errors,
        'warnings': warnings
    } 