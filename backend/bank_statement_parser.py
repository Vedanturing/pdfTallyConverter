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
    # Common header patterns for different banks
    BANK_PATTERNS = {
        'HDFC': {
            'date': r'Date|Txn Date|Transaction Date',
            'description': r'Narration|Description|Particulars',
            'debit': r'Debit|Withdrawal \(Dr\)|DR',
            'credit': r'Credit|Deposit \(Cr\)|CR',
            'balance': r'Balance|Running Balance|Closing Balance'
        },
        'ICICI': {
            'date': r'Date|Transaction Date|Tran Date',
            'description': r'Description|Narration|Particulars',
            'debit': r'Debit|Withdrawal|DR',
            'credit': r'Credit|Deposit|CR',
            'balance': r'Balance|Running Balance'
        },
        'SBI': {
            'date': r'Txn Date|Date|Value Date',
            'description': r'Description|Particulars|Narration',
            'debit': r'Debit|Withdrawal|DR',
            'credit': r'Credit|Deposit|CR',
            'balance': r'Balance|Running Balance'
        },
        'AXIS': {
            'date': r'Date|Transaction Date|Tran Date',
            'description': r'Particulars|Narration|Description',
            'debit': r'Debit|DR|Withdrawal',
            'credit': r'Credit|CR|Deposit',
            'balance': r'Balance|Running Balance'
        }
    }

    def __init__(self):
        self.detected_bank = None
        self.column_mapping = {}
        self.is_encrypted = False
        self.requires_password = False

    def _detect_bank_format(self, text: str) -> Optional[str]:
        """Detect bank format based on header patterns"""
        for bank, patterns in self.BANK_PATTERNS.items():
            matches = 0
            for field, pattern in patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    matches += 1
            if matches >= 3:  # At least 3 matching patterns
                return bank
        return None

    def _check_pdf_encryption(self, pdf_path: str) -> Tuple[bool, bool]:
        """Check if PDF is encrypted and needs password"""
        try:
            # Try PyMuPDF first
            doc = fitz.open(pdf_path)
            is_encrypted = doc.is_encrypted
            needs_password = is_encrypted and not doc.authenticate("")
            doc.close()
            return is_encrypted, needs_password
        except Exception:
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf = PdfReader(file)
                    is_encrypted = pdf.is_encrypted
                    needs_password = is_encrypted
                    return is_encrypted, needs_password
            except Exception as e:
                logger.error(f"Error checking PDF encryption: {e}")
                return False, False

    def _decrypt_pdf(self, pdf_path: str, password: str) -> Optional[io.BytesIO]:
        """Decrypt PDF with password and return as buffer"""
        try:
            # Try PyMuPDF first
            doc = fitz.open(pdf_path)
            if doc.is_encrypted and not doc.authenticate(password):
                raise PDFPasswordError("Invalid password")
            
            # Create temporary buffer
            buffer = io.BytesIO()
            pdf_writer = fitz.open()
            pdf_writer.insert_pdf(doc)
            pdf_writer.save(buffer)
            buffer.seek(0)
            doc.close()
            pdf_writer.close()
            return buffer
        except Exception as e:
            logger.error(f"Error decrypting PDF with PyMuPDF: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf = PdfReader(file)
                    if pdf.is_encrypted:
                        if not pdf.decrypt(password):
                            raise PDFPasswordError("Invalid password")
                    
                    # Create temporary buffer
                    buffer = io.BytesIO()
                    from PyPDF2 import PdfWriter
                    writer = PdfWriter()
                    for page in pdf.pages:
                        writer.add_page(page)
                    writer.write(buffer)
                    buffer.seek(0)
                    return buffer
            except Exception as e:
                logger.error(f"Error decrypting PDF with PyPDF2: {e}")
                raise PDFPasswordError(str(e))

    def _extract_text_with_pdfplumber(self, pdf_path: str, password: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber with password support"""
        try:
            # Check if PDF is encrypted
            is_encrypted, needs_password = self._check_pdf_encryption(pdf_path)
            self.is_encrypted = is_encrypted
            self.requires_password = needs_password

            if needs_password and not password:
                raise PDFPasswordError("PDF is password protected. Please provide password.")

            # If password protected, decrypt first
            if needs_password and password:
                pdf_buffer = self._decrypt_pdf(pdf_path, password)
                if not pdf_buffer:
                    raise PDFPasswordError("Failed to decrypt PDF")
                pdf_file = pdf_buffer
            else:
                pdf_file = pdf_path

            with pdfplumber.open(pdf_file, password=password if needs_password else "") as pdf:
                all_tables = []
                for page in pdf.pages:
                    # Try to find tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            if table and len(table) > 1:  # Has headers and data
                                all_tables.extend(table)
                    else:
                        # If no tables found, try to extract text with layout
                        text = page.extract_text()
                        if text:
                            lines = text.split('\n')
                            structured_lines = []
                            for line in lines:
                                parts = re.split(r'\s{2,}|\t|,(?=\s)', line)
                                if len(parts) >= 4:  # Minimum columns needed
                                    structured_lines.append(parts)
                            if structured_lines:
                                all_tables.extend(structured_lines)
                return all_tables
        except Exception as e:
            logger.error(f"Error in pdfplumber extraction: {e}")
            raise

    def _extract_text_with_fitz(self, pdf_path: str, password: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract text using PyMuPDF with password support"""
        try:
            doc = fitz.open(pdf_path)
            if doc.is_encrypted:
                if not password:
                    raise PDFPasswordError("PDF is password protected. Please provide password.")
                if not doc.authenticate(password):
                    raise PDFPasswordError("Invalid password")

            all_lines = []
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                lines = []
                current_line = []
                current_y = None
                
                for block in blocks:
                    if block.get("lines"):
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    y = span["origin"][1]
                                    if current_y is None:
                                        current_y = y
                                    elif abs(y - current_y) > 5:
                                        if current_line:
                                            lines.append(current_line)
                                        current_line = []
                                        current_y = y
                                    current_line.append(text)
                
                if current_line:
                    lines.append(current_line)
                all_lines.extend(lines)
            
            doc.close()
            return all_lines
        except Exception as e:
            logger.error(f"Error in fitz extraction: {e}")
            raise

    def _process_scanned_page(self, page_image: Image.Image) -> List[str]:
        """Process scanned page using OCR"""
        # Enhance image for better OCR
        img_cv = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Convert back to PIL Image
        img_pil = Image.fromarray(img_thresh)
        
        # Perform OCR
        text = pytesseract.image_to_string(img_pil, lang='eng')
        
        # Split into lines and clean
        lines = text.split('\n')
        return [line.strip() for line in lines if line.strip()]

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to YYYY-MM-DD format"""
        try:
            # Common date formats
            formats = [
                "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
                "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
                "%d-%m-%y", "%d/%m/%y", "%d.%m.%y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            
            # Try parsing with dateutil as fallback
            from dateutil import parser
            return parser.parse(date_str).strftime("%Y-%m-%d")
        except Exception:
            return date_str

    def _normalize_amount(self, amount_str: str) -> float:
        """Normalize amount to float"""
        if not amount_str or amount_str.strip() in ['-', 'NA', 'N/A']:
            return 0.0
        
        # Remove currency symbols and commas
        amount_str = re.sub(r'[₹,]', '', str(amount_str))
        # Extract numbers and decimal point
        amount_str = re.sub(r'[^\d.-]', '', amount_str)
        
        try:
            return float(amount_str)
        except ValueError:
            return 0.0

    def _map_columns(self, headers: List[str]) -> Dict[str, int]:
        """Map column headers to standardized fields"""
        column_map = {}
        headers_str = ' '.join(headers).upper()
        
        # Detect bank format if not already detected
        if not self.detected_bank:
            self.detected_bank = self._detect_bank_format(headers_str)
        
        if not self.detected_bank:
            # Generic mapping based on common patterns
            patterns = {
                'date': r'DATE|TXN|TRANSACTION',
                'description': r'NARRATION|DESCRIPTION|PARTICULARS',
                'debit': r'DEBIT|WITHDRAWAL|DR',
                'credit': r'CREDIT|DEPOSIT|CR',
                'balance': r'BALANCE'
            }
        else:
            patterns = self.BANK_PATTERNS[self.detected_bank]
        
        # Map columns based on patterns
        for idx, header in enumerate(headers):
            header_upper = header.upper()
            for field, pattern in patterns.items():
                if re.search(pattern, header_upper, re.IGNORECASE):
                    column_map[field] = idx
                    break
        
        return column_map

    def parse_statement(self, pdf_path: str, password: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main method to parse bank statement with password support"""
        try:
            # Check if PDF is encrypted
            is_encrypted, needs_password = self._check_pdf_encryption(pdf_path)
            self.is_encrypted = is_encrypted
            self.requires_password = needs_password

            if needs_password and not password:
                return pd.DataFrame(), {
                    "error": "PDF is password protected",
                    "requires_password": True
                }

            # Try pdfplumber first
            try:
                tables = self._extract_text_with_pdfplumber(pdf_path, password)
            except PDFPasswordError:
                return pd.DataFrame(), {
                    "error": "Invalid password",
                    "requires_password": True
                }
            except Exception as e:
                tables = []
                logger.error(f"pdfplumber extraction failed: {e}")

            if not tables:
                # Fall back to PyMuPDF
                try:
                    tables = self._extract_text_with_fitz(pdf_path, password)
                except PDFPasswordError:
                    return pd.DataFrame(), {
                        "error": "Invalid password",
                        "requires_password": True
                    }
                except Exception as e:
                    logger.error(f"fitz extraction failed: {e}")

            if not tables:
                # Check if PDF is scanned
                try:
                    doc = fitz.open(pdf_path)
                    if needs_password:
                        if not doc.authenticate(password):
                            raise PDFPasswordError("Invalid password")
                    
                    for page in doc:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        lines = self._process_scanned_page(img)
                        if lines:
                            tables.extend([line.split() for line in lines])
                    doc.close()
                except Exception as e:
                    logger.error(f"OCR processing failed: {e}")

            if not tables:
                return pd.DataFrame(), {
                    "error": "Unable to extract data from PDF",
                    "requires_manual": True
                }

            # Process extracted data
            headers = [str(cell).strip() for cell in tables[0]]
            self.column_mapping = self._map_columns(headers)
            
            if not self.column_mapping:
                return pd.DataFrame(), {
                    "error": "Unable to auto-detect format",
                    "requires_manual": True,
                    "headers": headers
                }

            # Process rows
            rows = []
            for row in tables[1:]:
                if not row or all(not cell for cell in row):
                    continue
                
                processed_row = {
                    'date': '',
                    'description': '',
                    'debit': 0.0,
                    'credit': 0.0,
                    'balance': 0.0
                }
                
                try:
                    for field, idx in self.column_mapping.items():
                        if idx < len(row):
                            value = str(row[idx]).strip()
                            if field == 'date':
                                processed_row[field] = self._normalize_date(value)
                            elif field in ['debit', 'credit', 'balance']:
                                processed_row[field] = self._normalize_amount(value)
                            else:
                                processed_row[field] = value
                    
                    if processed_row['date']:  # Only add rows with valid dates
                        rows.append(processed_row)
                except Exception as e:
                    logger.warning(f"Error processing row: {e}")
                    continue
            
            df = pd.DataFrame(rows)
            
            # Add metadata
            metadata = {
                "bank_detected": self.detected_bank,
                "column_mapping": self.column_mapping,
                "total_rows": len(df),
                "date_range": {
                    "start": df['date'].min() if not df.empty else None,
                    "end": df['date'].max() if not df.empty else None
                }
            }
            
            return df, metadata
            
        except PDFPasswordError as e:
            return pd.DataFrame(), {
                "error": str(e),
                "requires_password": True
            }
        except Exception as e:
            logger.error(f"Error parsing bank statement: {e}")
            return pd.DataFrame(), {
                "error": str(e),
                "requires_manual": True
            }

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