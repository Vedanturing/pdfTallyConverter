from fastapi import FastAPI, UploadFile, HTTPException, File, Request, Query, Body, Form, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.middleware.gzip import GZipMiddleware
import os
import shutil
import uuid
import pdfplumber
import pandas as pd
import numpy as np
import cv2
import pytesseract
from PIL import Image
import io
from typing import List, Dict, Optional, Any, Union
import xml.etree.ElementTree as ET
from datetime import datetime
import aiofiles
import json
from pydantic import BaseModel
import asyncio
from concurrent.futures import ProcessPoolExecutor
import logging
import sys
import traceback
import  uvicorn
import mimetypes
from fastapi import BackgroundTasks
from pathlib import Path
import re
from validation_utils import validate_table_data, get_validation_summary, validate_financial_data, ValidationError
from bank_statement_parser import process_bank_statement
from bank_matcher import BankMatcher
from gst_helper import GSTHelper
import fitz
from PyPDF2 import PdfReader
from audit_logger import audit_logger
import multiprocessing
from cachetools import TTLCache
import hashlib
from functools import lru_cache, wraps
import psutil
import threading
import time
from tally_parser import tally_parser
from collections import defaultdict
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Performance optimization - file processing cache
file_cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour

def get_file_hash(file_path: str) -> str:
    """Generate a hash for file content to use as cache key"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()[:16]  # Use first 16 chars for shorter keys

def cached_file_processing(func):
    """Decorator for caching file processing results"""
    @wraps(func)
    def wrapper(file_path: str, *args, **kwargs):
        try:
            # Create cache key from file hash and function args
            file_hash = get_file_hash(file_path)
            cache_key = f"{func.__name__}_{file_hash}_{str(args)}_{str(sorted(kwargs.items()))}"
            
            # Check cache first
            if cache_key in file_cache:
                logger.info(f"Cache hit for {func.__name__}: {file_path}")
                return file_cache[cache_key]
            
            # Process file if not in cache
            result = func(file_path, *args, **kwargs)
            
            # Store in cache
            file_cache[cache_key] = result
            logger.info(f"Cached result for {func.__name__}: {file_path}")
            
            return result
        except Exception as e:
            logger.error(f"Error in cached processing: {str(e)}")
            # Fallback to direct function call if caching fails
            return func(file_path, *args, **kwargs)
    
    return wrapper

app = FastAPI(
    title="PDF Tally Converter API",
    description="API for converting PDF files with user management",
    version="2.0.0"
)

# Add CORS middleware before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:3000"
    ],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1024)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get the start timestamp
    start_time = datetime.now()
    
    # Process the request
    response = await call_next(request)
    
    # Log the request after it's processed
    duration = (datetime.now() - start_time).total_seconds()
    
    # Only log API endpoints, skip static files
    if request.url.path.startswith("/api") or request.url.path in ["/upload", "/convert", "/validate"]:
        audit_logger.log_action(
            action_type="api_request",
            summary=f"{request.method} {request.url.path}",
            metadata={
                "duration": duration,
                "status_code": response.status_code,
                "client_host": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
    
    return response

@app.get("/test")
async def test():
    """Test endpoint to verify the API is working"""
    return {"message": "API is working!"}

@app.get("/debug")
async def debug(request: Request):
    """Debug endpoint to show environment information"""
    return {
        "cwd": os.getcwd(),
        "files_in_cwd": os.listdir(),
        "python_path": sys.path,
        "env_vars": dict(os.environ),
        "request_url": str(request.url),
        "request_headers": dict(request.headers)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {
        "message": "PDF Tally Converter API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "available_endpoints": [
            "/",
            "/test",
            "/debug",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "cwd": os.getcwd(),
            "python_version": sys.version
        }
    }

# Create necessary directories
UPLOAD_DIR = "uploads"
CONVERTED_DIR = "converted"
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file handling
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CONVERTED_DIR, exist_ok=True)

# Initialize process pool for CPU-intensive tasks with optimal worker count
MAX_WORKERS = min(32, (multiprocessing.cpu_count() or 1) + 4)
process_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)

# Initialize bank matcher
bank_matcher = BankMatcher()

# Initialize GST helper
gst_helper = GSTHelper()

# Import the hybrid parser
try:
    from hybrid_bank_parser import parse_with_hybrid_engine, HybridBankParser
    HYBRID_PARSER_AVAILABLE = True
    logger.info("Hybrid bank parser loaded successfully")
except ImportError as e:
    HYBRID_PARSER_AVAILABLE = False
    logger.warning(f"Hybrid bank parser not available: {str(e)}")

class ConversionError(Exception):
    pass

class CellMetadata(BaseModel):
    error: Optional[bool] = None
    confidence: Optional[float] = None
    status: Optional[str] = None
    originalValue: Optional[Any] = None

class TableCell(BaseModel):
    value: Any
    metadata: CellMetadata

class TableRow(BaseModel):
    cells: Dict[str, TableCell]

    class Config:
        arbitrary_types_allowed = True

class TableData(BaseModel):
    headers: List[str]
    rows: List[TableRow]

    class Config:
        arbitrary_types_allowed = True

class EditHistory(BaseModel):
    timestamp: int
    rowId: str
    columnKey: str
    oldValue: Any
    newValue: Any

class SavePayload(BaseModel):
    fileId: str
    originalData: TableData
    modifiedData: TableData
    editHistory: List[EditHistory]

class ConvertRequest(BaseModel):
    formats: List[str] = ["xlsx", "csv", "xml"]
    return_data: bool = False

class UserInDB(BaseModel):
    username: str
    email: Optional[str] = None
    disabled: Optional[bool] = None

async def save_upload_file(upload_file: UploadFile) -> tuple[str, str]:
    """Save uploaded file and return the file path with optimized streaming"""
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(upload_file.filename or "")[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    try:
        # Use larger chunk size for better performance
        OPTIMIZED_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            while True:
                chunk = await upload_file.read(OPTIMIZED_CHUNK_SIZE)
                if not chunk:
                    break
                await out_file.write(chunk)
        
        return file_path, file_id
    except Exception as e:
        # Clean up the file if there's an error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@cached_file_processing
def parse_tally_file(file_path: str) -> pd.DataFrame:
    """Parse Tally XML or TXT files"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.xml':
            logger.info(f"Parsing Tally XML file: {file_path}")
            return tally_parser.parse_xml_file(file_path)
        elif file_ext == '.txt':
            logger.info(f"Parsing Tally TXT file: {file_path}")
            return tally_parser.parse_txt_file(file_path)
        else:
            raise ValueError(f"Unsupported Tally file format: {file_ext}")
            
    except Exception as e:
        logger.error(f"Error parsing Tally file {file_path}: {str(e)}")
        raise

@cached_file_processing
def extract_tables_from_pdf(file_path: str, use_hybrid: bool = True) -> pd.DataFrame:
    """Extract tables from PDF using hybrid parser with fallback to existing parser"""
    logger.info(f"Extracting tables from PDF: {file_path} (hybrid={use_hybrid})")
    
    if not os.path.exists(file_path):
        logger.error(f"PDF file not found: {file_path}")
        return pd.DataFrame()
    
    # Try hybrid parser first if available and enabled
    if use_hybrid and HYBRID_PARSER_AVAILABLE:
        try:
            logger.info("Using hybrid bank parser")
            df = parse_with_hybrid_engine(file_path, enable_advanced=True)
            if df is not None and not df.empty:
                logger.info(f"Hybrid parser successful, extracted {len(df)} rows")
                return df
            else:
                logger.warning("Hybrid parser returned empty result, falling back to original")
        except Exception as e:
            logger.error(f"Hybrid parser failed: {str(e)}, falling back to original parser")
    
    # Fallback to original parser logic
    logger.info("Using original extraction logic")
    return _extract_tables_original(file_path)

def _extract_tables_original(file_path: str) -> pd.DataFrame:
    """Original table extraction logic"""
    try:
        # First try to validate PDF file
        try:
            with open(file_path, 'rb') as f:
                # Try PyMuPDF first
                try:
                    doc = fitz.open(file_path)
                    if doc.page_count == 0:
                        raise ValueError("PDF has no pages")
                    doc.close()
                except Exception as e:
                    logger.warning(f"PyMuPDF validation failed: {str(e)}")
                    # Try PyPDF2 as fallback
                    reader = PdfReader(f)
                    if len(reader.pages) == 0:
                        raise ValueError("PDF has no pages")
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return pd.DataFrame()

        # Try general table extraction first
        logger.info("Starting pdfplumber table extraction...")
        try:
            with pdfplumber.open(file_path) as pdf:
                all_tables = []
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num} of {len(pdf.pages)}")
                    
                    # Try different table finding strategies
                    table_settings = [
                        {"vertical_strategy": "text", "horizontal_strategy": "text"},
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
                        {
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "text_tolerance": 3,
                            "text_x_tolerance": 3,
                            "intersection_tolerance": 3,
                            "snap_tolerance": 3,
                            "join_tolerance": 3,
                        }
                    ]
                    
                    tables = []
                    for settings in table_settings:
                        if not tables:  # Only try next strategy if no tables found
                            try:
                                found_tables = page.find_tables(settings)
                                if found_tables:
                                    tables = [table.extract() for table in found_tables]
                                    if tables and any(table for table in tables):
                                        logger.info(f"Found {len(tables)} tables with settings: {settings}")
                                        break
                            except Exception as e:
                                logger.warning(f"Table extraction failed with settings {settings}: {str(e)}")
                    
                    if tables:
                        # Filter out None tables and empty tables
                        valid_tables = [table for table in tables if table and len(table) > 1]
                        all_tables.extend(valid_tables)
                    else:
                        logger.warning(f"No tables found on page {page_num}")
                
                if all_tables:
                    logger.info(f"Found {len(all_tables)} tables total")
                    # Use the first table's headers
                    first_table = all_tables[0]
                    if not first_table or len(first_table) == 0:
                        logger.warning("First table is empty")
                        return pd.DataFrame()
                    
                    headers = first_table[0]
                    if not headers or not any(headers):
                        logger.warning("No valid headers found in first table")
                        return pd.DataFrame()
                    
                    # Combine all rows from all tables
                    rows = []
                    for table in all_tables:
                        if table and len(table) > 1:
                            rows.extend(table[1:])  # Skip header row for subsequent tables
                    
                    if not rows:
                        logger.warning("No data rows found in tables")
                        return pd.DataFrame()
                    
                    logger.info(f"Creating DataFrame with {len(headers)} columns and {len(rows)} rows")
                    df = pd.DataFrame(rows, columns=headers)
                    df = df.fillna('')  # Replace NaN with empty string
                    
                    # Clean up the data
                    df = df.replace(r'^\s*$', '', regex=True)  # Replace whitespace-only cells
                    df = df.replace(r'\s+', ' ', regex=True)   # Normalize whitespace
                    
                    # Try to detect if this is tabular data
                    if len(df.columns) >= 3 and len(df) > 0:
                        logger.info("Valid tabular data found, processing columns...")
                        # Standardize column names
                        column_mapping = {
                            col: col.lower().replace(' ', '_') if col else f'col_{i}'
                            for i, col in enumerate(df.columns)
                        }
                        df = df.rename(columns=column_mapping)
                        
                        # Ensure required columns exist
                        required_columns = {
                            'date': ['date', 'tran_date', 'transaction_date', 'value_date'],
                            'narration': ['narration', 'description', 'particulars', 'details'],
                            'amount': ['amount', 'debit', 'credit', 'dr', 'cr'],
                            'balance': ['balance', 'running_balance', 'closing_balance']
                        }
                        
                        # Map existing columns to standard names
                        for std_col, variants in required_columns.items():
                            if std_col not in df.columns:
                                # Find first matching variant
                                for var in variants:
                                    if var in df.columns:
                                        df = df.rename(columns={var: std_col})
                                        break
                                # If no variant found, add empty column
                                if std_col not in df.columns:
                                    df[std_col] = ''
                        
                        # Clean up date column
                        if 'date' in df.columns:
                            try:
                                df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
                                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                            except Exception as e:
                                logger.warning(f"Error processing date column: {str(e)}")
                        
                        # Clean up amount columns
                        for col in ['amount', 'balance']:
                            if col in df.columns:
                                try:
                                    df[col] = df[col].apply(lambda x: str(x).replace('₹', '').replace(',', '').strip())
                                    # Only convert to numeric if the value is not empty, otherwise keep as string
                                    df[col] = df[col].apply(lambda x: pd.to_numeric(x, errors='coerce') if x and x != '' else x)
                                except Exception as e:
                                    logger.warning(f"Error processing {col} column: {str(e)}")
                        
                        logger.info(f"Successfully processed tabular data with shape: {df.shape}")
                        return df
                    else:
                        logger.warning("Extracted data doesn't look like valid tabular data")
        except Exception as e:
            logger.error(f"pdfplumber table extraction failed: {str(e)}")
            logger.error(traceback.format_exc())
            
        # If general table extraction fails or doesn't look like tabular data,
        # try processing as bank statement
        logger.info("Trying bank statement parser as fallback...")
        try:
            df = process_bank_statement(file_path)
            if df is not None and not df.empty:
                logger.info("Successfully extracted data using bank statement parser")
                return df
            else:
                logger.warning("Bank statement parser returned empty DataFrame")
        except Exception as e:
            logger.error(f"Bank statement parser failed: {str(e)}")
            logger.error(traceback.format_exc())
        
        # If both methods fail, return empty DataFrame
        logger.warning("All extraction methods failed, returning empty DataFrame")
        return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in table extraction: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

@cached_file_processing
def process_image_ocr(file_path: str) -> pd.DataFrame:
    """Process image using OCR to extract tabular data with fallback when tesseract is not available"""
    logger.info(f"Processing image using OCR: {file_path}")
    
    try:
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"Failed to read image file: {file_path}")
            raise ConversionError("Failed to read image file")

        logger.info(f"Image loaded successfully with shape: {img.shape}")
        
        # First try using tesseract if available
        try:
            import pytesseract
            logger.info("Using Tesseract OCR")
            return _process_with_tesseract(img)
        except ImportError:
            logger.warning("Tesseract not available, using alternative OCR")
        except Exception as e:
            if "tesseract is not installed" in str(e).lower():
                logger.warning("Tesseract not installed, using alternative OCR")
            else:
                logger.error(f"Tesseract failed: {str(e)}, using alternative OCR")
        
        # Try hybrid parser
        if HYBRID_PARSER_AVAILABLE:
            try:
                logger.info("Using hybrid parser for image")
                from hybrid_bank_parser import HybridBankParser
                parser = HybridBankParser(enable_advanced_detection=True)
                
                # Extract cells directly from image
                from hybrid_bank_parser import TableRegion
                region = TableRegion(0, 0, img.shape[1], img.shape[0], 1.0, 0)
                cells = parser.extract_cells_from_region(img, region)
                
                if cells:
                    parsed_table = parser.reconstruct_table(cells)
                    parsed_table = parser.map_to_schema(parsed_table)
                    parsed_table = parser.clean_data(parsed_table)
                    
                    if parsed_table.rows:
                        df = pd.DataFrame(parsed_table.rows, columns=parsed_table.headers)
                        logger.info(f"Hybrid parser successful: {df.shape}")
                        return df
            except Exception as e:
                logger.warning(f"Hybrid parser failed: {str(e)}")
        
        # Use simple OCR fallback
        try:
            logger.info("Using simple OCR fallback")
            from simple_ocr_fallback import extract_with_simple_ocr
            df = extract_with_simple_ocr(img)
            if not df.empty:
                logger.info(f"Simple OCR successful: {df.shape}")
                return df
        except Exception as e:
            logger.warning(f"Simple OCR fallback failed: {str(e)}")
        
        # Final fallback: return structured dummy data to indicate we found the image
        logger.warning("All OCR methods failed, returning placeholder data")
        return pd.DataFrame({
            'date': [datetime.now().strftime('%Y-%m-%d')],
            'description': ['Image processed - OCR unavailable'],
            'amount': [0.0],
            'balance': [0.0]
        })
        
    except ConversionError:
        raise
    except Exception as e:
        logger.error(f"Error in OCR processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise ConversionError(f"Failed to process image: {str(e)}")

def _process_with_tesseract(img: np.ndarray) -> pd.DataFrame:
    """Process image using Tesseract OCR"""
    import pytesseract
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Improve image quality for OCR
    # Apply threshold to get better text recognition
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR with different configurations
    custom_config = r'--oem 3 --psm 6'  # Treat the image as a single uniform block of text
    
    try:
        text = pytesseract.image_to_string(thresh, config=custom_config)
        logger.info(f"Tesseract OCR completed, extracted text length: {len(text)}")
    except Exception as e:
        logger.warning(f"Tesseract with custom config failed: {str(e)}, trying default config")
        text = pytesseract.image_to_string(thresh)
    
    if not text or not text.strip():
        logger.error("No text detected in image")
        return pd.DataFrame()
    
    # Convert OCR text to DataFrame
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if not lines:
        logger.error("No valid lines found in OCR text")
        return pd.DataFrame()
        
    logger.info(f"Found {len(lines)} lines of text")
    
    # Try to detect table structure
    # Look for lines that might be headers (containing common table keywords)
    header_keywords = ['date', 'amount', 'description', 'balance', 'debit', 'credit', 'transaction']
    header_line_idx = -1
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in header_keywords):
            # Check if this line has multiple "columns" (spaces between words)
            words = line.split()
            if len(words) >= 3:  # At least 3 columns
                header_line_idx = i
                logger.info(f"Found potential header line at index {i}: {line}")
                break
    
    if header_line_idx == -1:
        # If no clear header found, assume first line is header
        header_line_idx = 0
        logger.warning("No clear header line found, using first line as header")
    
    # Extract headers and data
    headers = lines[header_line_idx].split()
    data_lines = lines[header_line_idx + 1:]
    
    if not headers:
        logger.error("No headers found")
        return pd.DataFrame()
    
    if not data_lines:
        logger.error("No data lines found")
        return pd.DataFrame()
    
    logger.info(f"Headers: {headers}")
    logger.info(f"Data lines count: {len(data_lines)}")
    
    # Convert data lines to rows
    data = []
    for line in data_lines:
        if line.strip():
            # Split line into fields
            # Try to split by whitespace but handle cases where values might contain spaces
            words = line.split()
            if len(words) >= len(headers):
                # If we have more words than headers, try to group them intelligently
                if len(words) > len(headers):
                    # Keep first and last fields, combine middle ones
                    row = [words[0]]  # First field (likely date)
                    if len(headers) > 2:
                        # Combine middle fields (likely description)
                        middle_words = words[1:-len(headers)+2]
                        row.append(' '.join(middle_words))
                        # Add remaining fields
                        row.extend(words[-len(headers)+2:])
                    else:
                        row.extend(words[1:len(headers)])
                else:
                    row = words
                
                # Pad or trim row to match header count
                while len(row) < len(headers):
                    row.append('')
                row = row[:len(headers)]
                
                data.append(row)
            else:
                logger.warning(f"Skipping line with insufficient columns: {line}")
    
    if not data:
        logger.error("No valid data rows found")
        return pd.DataFrame()
    
    logger.info(f"Creating DataFrame with {len(headers)} columns and {len(data)} rows")
    df = pd.DataFrame(data, columns=headers)
    
    # Clean up column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Basic data cleaning
    df = df.fillna('')
    df = df.replace(r'^\s*$', '', regex=True)
    
    logger.info(f"Successfully created DataFrame from Tesseract OCR with shape: {df.shape}")
    return df

def create_tally_xml(df: pd.DataFrame, output_path: str) -> None:
    """Create Tally XML from DataFrame"""
    logger.info(f"Creating Tally XML at: {output_path}")
    try:
        # Create the root element
        root = ET.Element("ENVELOPE")
        
        # Add header
        header = ET.SubElement(root, "HEADER")
        ET.SubElement(header, "TALLYREQUEST").text = "Import Data"
        
        # Add body
        body = ET.SubElement(root, "BODY")
        importdata = ET.SubElement(body, "IMPORTDATA")
        requestdesc = ET.SubElement(importdata, "REQUESTDESC")
        reportname = ET.SubElement(requestdesc, "REPORTNAME").text = "Vouchers"
        
        # Add data
        requestdata = ET.SubElement(importdata, "REQUESTDATA")
        
        # Convert each row to XML
        for _, row in df.iterrows():
            voucher = ET.SubElement(requestdata, "VOUCHER")
            for col in df.columns:
                if pd.notna(row[col]):  # Only add non-null values
                    ET.SubElement(voucher, col.upper().replace(" ", "")).text = str(row[col])
        
        # Create the XML file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        
        logger.info("Successfully created Tally XML file")
        
    except Exception as e:
        logger.error(f"Error creating Tally XML: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def convert_file(file_path: str, file_id: str, output_formats: List[str] = ["xlsx", "csv", "xml"]) -> Dict:
    """Convert file to multiple formats with enhanced processing"""
    results = {}
    
    try:
        logger.info(f"Starting conversion for file: {file_path}")
        
        # Parse file and get data
        if file_path.endswith('.xml') or file_path.endswith('.txt'):
            df = parse_tally_file(file_path)
        elif file_path.endswith('.pdf'):
            df = extract_tables_from_pdf(file_path)
        else:
            df = process_image_ocr(file_path)
        
        if df is None or df.empty:
            logger.warning("No data extracted from file")
            results['error'] = "No data could be extracted from the file"
            return results
        
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # Enhanced amount column processing
        def enhance_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
            """Enhance amount columns in the DataFrame"""
            try:
                # Create a copy to avoid modifying the original
                df = df.copy()
                
                # Identify potential amount columns using keywords
                amount_keywords = ['amount', 'amt', 'balance', 'total', 'sum', 'value', 'money',
                                 'debit', 'credit', 'withdrawal', 'deposit', 'rupees', 'rs', 'inr']
                
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(keyword in col_lower for keyword in amount_keywords):
                        try:
                            # Try to convert to numeric, replacing errors with NaN
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            continue
                
                return df
                
            except Exception as e:
                logger.error(f"Error enhancing amount columns: {e}")
                return df  # Return original DataFrame if enhancement fails
        
        # Apply amount enhancement
        df = enhance_amount_columns(df)

        # Create output directory
        converted_dir = "converted"
        ensure_directory(converted_dir)

        # Convert to different formats
        base_filename = f"{file_id}"
        
        for format in output_formats:
            try:
                if format == "xlsx":
                    output_path = os.path.join(converted_dir, f"{base_filename}.xlsx")
                    df.to_excel(output_path, index=False, engine='openpyxl')
                    results['xlsx'] = output_path
                    logger.info(f"Excel file created: {output_path}")
                elif format == "csv":
                    output_path = os.path.join(converted_dir, f"{base_filename}.csv")
                    df.to_csv(output_path, index=False, encoding='utf-8')
                    results['csv'] = output_path
                    logger.info(f"CSV file created: {output_path}")
                elif format == "json":
                    output_path = os.path.join(converted_dir, f"{base_filename}.json")
                    json_data = {
                        "headers": df.columns.tolist(),
                        "rows": df.to_dict('records')
                    }
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
                    results['json'] = output_path
                    logger.info(f"JSON file created: {output_path}")
                elif format == "xml":
                    output_path = os.path.join(converted_dir, f"{base_filename}.xml")
                    create_tally_xml(df, output_path)
                    results['xml'] = output_path
                    logger.info(f"XML file created: {output_path}")
            except Exception as e:
                logger.error(f"Error creating {format} file: {str(e)}")
                results[f'{format}_error'] = str(e)
                
        # Save table data for preview
        table_data = df.to_dict('records')
        results['table_data'] = table_data
        results['headers'] = df.columns.tolist()
        
        logger.info(f"Conversion completed successfully. Formats: {list(results.keys())}")
        
        return results

    except Exception as e:
        logger.error(f"Error in convert_file: {str(e)}")
        logger.error(traceback.format_exc())
        results['error'] = str(e)
        return results

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    password: Optional[str] = Form(None)
):
    """
    Upload a file and process it
    """
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Get file extension
        filename = file.filename or ""  # Ensure filename is a string
        file_extension = os.path.splitext(filename)[1].lower()
        
        # Check if file type is supported
        supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.xml', '.txt']
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(supported_extensions)}"
            )
        
        # Create temporary file path
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{file_id}{file_extension}")
        
        # Create final file path
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        # Save uploaded file to temporary location
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate file is a PDF if it has .pdf extension
        if file_extension == '.pdf':
            try:
                # Try PyMuPDF first
                doc = None
                try:
                    doc = fitz.open(temp_path)
                    if doc.needs_pass:
                        if not password:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            return JSONResponse(
                                status_code=401,
                                content={
                                    "requires_password": True,
                                    "message": "PDF is password protected. Please provide a password."
                                }
                            )
                        if not doc.authenticate(password):
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            return JSONResponse(
                                status_code=401,
                                content={
                                    "requires_password": True,
                                    "message": "Incorrect password. Please try again."
                                }
                            )
                    # Validate PDF has pages
                    if doc.page_count == 0:
                        raise ValueError("PDF file has no pages")
                finally:
                    if doc:
                        doc.close()
                
            except Exception as e:
                logger.warning(f"PyMuPDF validation failed: {str(e)}, trying PyPDF2")
                # Try PyPDF2 as fallback
                try:
                    with open(temp_path, 'rb') as pdf_file:
                        pdf_reader = PdfReader(pdf_file)
                        if pdf_reader.is_encrypted:
                            if not password:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                return JSONResponse(
                                    status_code=401,
                                    content={
                                        "requires_password": True,
                                        "message": "PDF is password protected. Please provide a password."
                                    }
                                )
                            try:
                                pdf_reader.decrypt(password)
                            except:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                return JSONResponse(
                                    status_code=401,
                                    content={
                                        "requires_password": True,
                                        "message": "Incorrect password. Please try again."
                                    }
                                )
                        # Validate PDF has pages
                        if len(pdf_reader.pages) == 0:
                            raise ValueError("PDF file has no pages")
                except Exception as e:
                    logger.error(f"PDF validation failed: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid or corrupted PDF file"
                    )
        
        # If we get here, either the file is not a PDF or it's a valid PDF
        # Move the temp file to final location
        os.rename(temp_path, file_path)
        
        # Store file metadata including original filename
        try:
            from file_metadata import store_file_metadata
            if filename:  # Only store metadata if we have a filename
                store_file_metadata(file_id, filename)
                logger.info(f"Stored metadata for file: {filename}")
        except Exception as e:
            logger.warning(f"Failed to store file metadata: {e}")
        
        # Return success response
        return {
            "file_id": file_id,
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/convert/{file_id}")
async def convert_uploaded_file(
    file_id: str,
    request: ConvertRequest,
    current_user: Optional['UserInDB'] = Depends(lambda: None)
):
    """Convert a file and return the extracted data with progress tracking"""
    try:
        logger.info(f"Starting conversion for file_id: {file_id}")
        
        progress_tracker.update_progress(file_id, "starting", 5, "Finding uploaded file...")
        
        original_file = None
        for ext in ['.pdf', '.png', '.jpg', '.jpeg', '.xml', '.txt']:
            path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
            if os.path.exists(path):
                original_file = path
                break
        
        if not original_file:
            raise HTTPException(
                status_code=404,
                detail="Original file not found"
            )
        
        # Extract data based on file type
        try:
            if original_file.lower().endswith('.pdf'):
                df = extract_tables_from_pdf(original_file)
            elif original_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                df = process_image_ocr(original_file)
            elif original_file.lower().endswith(('.xml', '.txt')):
                df = parse_tally_file(original_file)
            else:
                raise ValueError(f"Unsupported file format: {original_file}")

            if df.empty:
                raise ValueError("No data could be extracted from the file")

            # Convert DataFrame to dictionary format
            headers = df.columns.tolist()
            rows = []
            for _, row in df.iterrows():
                row_dict = {}
                for col in headers:
                    value = row[col]
                    if pd.isna(value):
                        row_dict[col] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        row_dict[col] = float(value) if isinstance(value, np.floating) else int(value)
                    elif isinstance(value, pd.Timestamp):
                        row_dict[col] = value.strftime('%Y-%m-%d')
                    else:
                        row_dict[col] = str(value)
                rows.append(row_dict)

            # Save converted data
            os.makedirs(CONVERTED_DIR, exist_ok=True)
            json_data = {"headers": headers, "rows": rows}
            json_path = os.path.join(CONVERTED_DIR, f"{file_id}.json")
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
            
            # Try to extract bearer name
            try:
                from name_extractor import NameExtractor
                from file_metadata import update_extracted_name
                
                name_extractor = NameExtractor()
                extracted_name = None
                
                if original_file.lower().endswith('.pdf'):
                    extracted_name = name_extractor.extract_name_from_pdf(original_file)
                elif original_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    extracted_name = name_extractor.extract_name_from_image(original_file)
                
                if extracted_name:
                    update_extracted_name(file_id, extracted_name)
                    logger.info(f"Extracted name for {file_id}: {extracted_name}")
            except Exception as e:
                logger.warning(f"Failed to extract bearer name for {file_id}: {e}")

            response = {
                "file_id": file_id,
                "status": "converted",
                "headers": headers,
                "row_count": len(rows),
                "download_links": {
                    "json": f"/api/download/{file_id}/json",
                    "xlsx": f"/api/download/{file_id}/xlsx",
                    "csv": f"/api/download/{file_id}/csv"
                }
            }
            
            # If return_data is requested, include the rows
            if request.return_data:
                response["rows"] = rows
                
            return response
            
        except Exception as e:
            logger.error(f"Conversion error for {file_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    except Exception as e:
        logger.error(f"Processing error for {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/download-multiple")
async def download_multiple_files(request: Request, background_tasks: BackgroundTasks):
    """Download multiple files as a ZIP"""
    try:
        data = await request.json()
        file_ids = data.get('file_ids', [])
        format_type = data.get('format', 'xlsx')
        
        if not file_ids:
            raise HTTPException(status_code=400, detail="No file IDs provided")
        
        import zipfile
        import io
        
        # Create a zip file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_id in file_ids:
                try:
                    # Generate individual file
                    if format_type == 'xlsx':
                        file_path = os.path.join(CONVERTED_DIR, f"{file_id}.xlsx")
                        if not os.path.exists(file_path):
                            # Generate from JSON
                            json_path = os.path.join(CONVERTED_DIR, f"{file_id}.json")
                            if os.path.exists(json_path):
                                with open(json_path, 'r') as f:
                                    json_data = json.load(f)
                                
                                if isinstance(json_data, dict) and 'rows' in json_data:
                                    df = pd.DataFrame(json_data['rows'])
                                else:
                                    df = pd.DataFrame(json_data)
                                
                                df.to_excel(file_path, index=False, engine='openpyxl')
                    
                    elif format_type == 'csv':
                        file_path = os.path.join(CONVERTED_DIR, f"{file_id}.csv")
                        if not os.path.exists(file_path):
                            json_path = os.path.join(CONVERTED_DIR, f"{file_id}.json")
                            if os.path.exists(json_path):
                                with open(json_path, 'r') as f:
                                    json_data = json.load(f)
                                
                                if isinstance(json_data, dict) and 'rows' in json_data:
                                    df = pd.DataFrame(json_data['rows'])
                                else:
                                    df = pd.DataFrame(json_data)
                                
                                df.to_csv(file_path, index=False)
                    
                    # Generate dynamic filename
                    try:
                        from name_extractor import generate_dynamic_filename
                        from file_metadata import get_file_metadata
                        
                        metadata = get_file_metadata(file_id)
                        original_filename = metadata.get('original_filename') if metadata else None
                        extracted_name = metadata.get('extracted_name') if metadata else None
                        
                        dynamic_filename = generate_dynamic_filename(
                            file_id=file_id,
                            original_filename=original_filename,
                            extracted_name=extracted_name,
                            file_format=format_type
                        )
                    except Exception as e:
                        logger.warning(f"Could not generate dynamic filename: {e}")
                        dynamic_filename = f"{file_id}.{format_type}"
                    
                    # Add file to zip
                    if os.path.exists(file_path):
                        zip_file.write(file_path, dynamic_filename)
                    
                except Exception as e:
                    logger.error(f"Error adding {file_id} to zip: {e}")
                    continue
        
        zip_buffer.seek(0)
        
        # Generate zip filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        zip_filename = f"converted_files_{timestamp}.zip"
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
        
    except Exception as e:
        logger.error(f"Multiple download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Progress tracking for conversions
class ConversionProgress:
    def __init__(self):
        self.progress_data = defaultdict(dict)
    
    def update_progress(self, file_id: str, stage: str, progress: int, message: str = ""):
        self.progress_data[file_id] = {
            "stage": stage,
            "progress": progress,
            "message": message,
            "timestamp": time.time()
        }
    
    def get_progress(self, file_id: str) -> Dict[str, Any]:
        return self.progress_data.get(file_id, {
            "stage": "pending",
            "progress": 0,
            "message": "",
            "timestamp": time.time()
        })
    
    def clear_progress(self, file_id: str):
        if file_id in self.progress_data:
            del self.progress_data[file_id]

# Global progress tracker
progress_tracker = ConversionProgress()

# Improved stream-based file processing
async def process_file_stream(file_path: str, file_id: str) -> pd.DataFrame:
    """Process file with streaming and progress tracking"""
    try:
        progress_tracker.update_progress(file_id, "initializing", 10, "Starting file processing...")
        
        # Check file extension and use appropriate processing
        if file_path.lower().endswith('.pdf'):
            progress_tracker.update_progress(file_id, "parsing", 20, "Reading PDF structure...")
            
            # Check if we can use text extraction first (faster than OCR)
            try:
                import fitz
                doc = fitz.open(file_path)
                text_content = ""
                for page_num in range(min(3, doc.page_count)):  # Check first 3 pages
                    page = doc.page(page_num)
                    text_content += page.get_text()
                
                # If we have substantial text, try table extraction first
                if len(text_content.strip()) > 100:
                    progress_tracker.update_progress(file_id, "text_extraction", 40, "Extracting text-based tables...")
                    df = await asyncio.get_event_loop().run_in_executor(
                        process_pool, _extract_tables_text_based, file_path
                    )
                    
                    if not df.empty:
                        progress_tracker.update_progress(file_id, "completed", 100, "Text extraction successful")
                        return df
                
                doc.close()
            except Exception as e:
                logger.warning(f"Text extraction failed, falling back to OCR: {e}")
            
            # Fallback to hybrid/OCR processing
            progress_tracker.update_progress(file_id, "ocr_processing", 60, "Using OCR for table extraction...")
            df = await asyncio.get_event_loop().run_in_executor(
                process_pool, extract_tables_from_pdf, file_path
            )
            
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            progress_tracker.update_progress(file_id, "image_processing", 30, "Processing image...")
            df = await asyncio.get_event_loop().run_in_executor(
                process_pool, process_image_ocr, file_path
            )
            
        elif file_path.lower().endswith(('.xml', '.txt')):
            progress_tracker.update_progress(file_id, "tally_parsing", 50, "Parsing Tally format...")
            df = await asyncio.get_event_loop().run_in_executor(
                process_pool, parse_tally_file, file_path
            )
            
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        progress_tracker.update_progress(file_id, "finalizing", 90, "Finalizing data...")
        
        if df.empty:
            raise ValueError("No data could be extracted from the file")
        
        progress_tracker.update_progress(file_id, "completed", 100, "Processing completed successfully")
        return df
        
    except Exception as e:
        progress_tracker.update_progress(file_id, "error", 0, f"Error: {str(e)}")
        raise

def _extract_tables_text_based(file_path: str) -> pd.DataFrame:
    """Extract tables from PDF using text-based approach"""
    try:
        doc = fitz.open(file_path)
        all_text = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            all_text.append(text)
            
        doc.close()
        
        # Process text and convert to DataFrame
        # ... rest of the function implementation ...
        
    except Exception as e:
        logger.error(f"Error in text-based extraction: {e}")
        return pd.DataFrame()

# Progress tracking endpoint
@app.get("/file/{file_id}")
async def get_file(file_id: str, convert: bool = False):
    """Get file content or converted data"""
    try:
        # First try to find the original file
        file_path = None
        for ext in ['.pdf', '.png', '.jpg', '.jpeg', '.xml', '.txt']:
            potential_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")
        
        if convert:
            # Return converted data if available
            json_path = os.path.join(CONVERTED_DIR, f"{file_id}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                return JSONResponse(content={
                    "success": True,
                    "data": data
                })
            else:
                # File exists but not converted yet
                return JSONResponse(content={
                    "success": False,
                    "message": "File not converted yet"
                })
        else:
            # Return the actual file
            def iterfile():
                with open(file_path, mode="rb") as file_like:
                    yield from file_like
            
            # Determine content type
            content_type = "application/octet-stream"
            if file_path.endswith('.pdf'):
                content_type = "application/pdf"
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                content_type = f"image/{file_path.split('.')[-1]}"
            
            return StreamingResponse(iterfile(), media_type=content_type)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reconcile")
async def reconcile_gst_data(file: UploadFile = File(...)):
    """
    GST Reconciliation endpoint - converts Excel macro-based reconciliation to Python
    Accepts Excel file with 'Books' and 'Portal' sheets for GST data reconciliation
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files are supported")
        
        # Save uploaded file
        file_path, file_id = await save_upload_file(file)
        
        logger.info(f"Starting GST reconciliation for file: {file.filename}")
        
        # Read Excel sheets
        try:
            books_df = pd.read_excel(file_path, sheet_name='Books')
            portal_df = pd.read_excel(file_path, sheet_name='Portal')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading Excel sheets: {str(e)}")
        
        # Validate required columns exist
        required_books_cols = ['GSTIN', 'Invoice_No', 'Invoice_Date', 'CGST', 'SGST', 'IGST']
        required_portal_cols = ['GSTIN', 'Invoice_No', 'Invoice_Date', 'CGST', 'SGST', 'IGST']
        
        missing_books = [col for col in required_books_cols if col not in books_df.columns]
        missing_portal = [col for col in required_portal_cols if col not in portal_df.columns]
        
        if missing_books or missing_portal:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing columns - Books: {missing_books}, Portal: {missing_portal}"
            )
        
        # Sanitize and prepare data
        def sanitize_dataframe(df):
            # Convert tax columns to float
            for col in ['CGST', 'SGST', 'IGST']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
            
            # Standardize date format
            df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'], errors='coerce')
            
            # Remove empty rows
            df = df.dropna(subset=['GSTIN', 'Invoice_No'])
            
            # Calculate total tax
            df['Total_Tax'] = df['CGST'] + df['SGST'] + df['IGST']
            
            return df
        
        books_clean = sanitize_dataframe(books_df.copy())
        portal_clean = sanitize_dataframe(portal_df.copy())
        
        # Step 1-2: Match entries by GSTIN + Invoice + Amount + Date
        def create_match_key(row):
            return f"{row['GSTIN']}_{row['Invoice_No']}_{row['Total_Tax']}_{row['Invoice_Date'].strftime('%Y-%m-%d') if pd.notna(row['Invoice_Date']) else 'NO_DATE'}"
        
        books_clean['match_key'] = books_clean.apply(create_match_key, axis=1)
        portal_clean['match_key'] = portal_clean.apply(create_match_key, axis=1)
        
        # Find matches
        books_matches = books_clean[books_clean['match_key'].isin(portal_clean['match_key'])].copy()
        portal_matches = portal_clean[portal_clean['match_key'].isin(books_clean['match_key'])].copy()
        
        # Find unmatched entries
        books_unmatched = books_clean[~books_clean['match_key'].isin(portal_clean['match_key'])].copy()
        portal_unmatched = portal_clean[~portal_clean['match_key'].isin(books_clean['match_key'])].copy()
        
        # Mark matched entries
        books_matches['Status'] = 'Matched'
        portal_matches['Status'] = 'Matched'
        books_unmatched['Status'] = 'Unmatched'
        portal_unmatched['Status'] = 'Unmatched'
        
        # Step 9: Generate pivot table of unmatched entries grouped by GSTIN
        unmatched_summary = books_unmatched.groupby('GSTIN').agg({
            'Invoice_No': 'count',
            'Total_Tax': 'sum',
            'CGST': 'sum',
            'SGST': 'sum',
            'IGST': 'sum'
        }).rename(columns={'Invoice_No': 'Unmatched_Count'})
        
        # Step 11-12: Generate final summary
        summary_stats = {
            'total_books_entries': len(books_clean),
            'total_portal_entries': len(portal_clean),
            'matched_entries': len(books_matches),
            'books_unmatched': len(books_unmatched),
            'portal_unmatched': len(portal_unmatched),
            'match_percentage': round((len(books_matches) / len(books_clean)) * 100, 2) if len(books_clean) > 0 else 0
        }
        
        # Create output workbook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"Reco_Report_{timestamp}.xlsx"
        output_path = os.path.join(CONVERTED_DIR, output_filename)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write matched entries
            books_matches.to_excel(writer, sheet_name='Books_Matched', index=False)
            portal_matches.to_excel(writer, sheet_name='Portal_Matched', index=False)
            
            # Write unmatched entries
            books_unmatched.to_excel(writer, sheet_name='Books_Unmatched', index=False)
            portal_unmatched.to_excel(writer, sheet_name='Portal_Unmatched', index=False)
            
            # Write summary pivot
            unmatched_summary.to_excel(writer, sheet_name='Unmatched_Summary')
            
            # Write overall summary
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_excel(writer, sheet_name='Overall_Summary', index=False)
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(f"GST reconciliation completed. Output: {output_filename}")
        
        return JSONResponse(content={
            "success": True,
            "message": "GST reconciliation completed successfully",
            "summary": summary_stats,
            "output_file": output_filename,
            "download_url": f"/download/{output_filename}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GST reconciliation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reconciliation failed: {str(e)}")

@app.get("/api/download/{file_id}/{format}")
async def download_converted_file(file_id: str, format: str):
    """Download converted file in specified format"""
    try:
        # Validate format
        if format not in ['json', 'xlsx', 'csv', 'xml']:
            raise HTTPException(status_code=400, detail="Invalid format. Supported: json, xlsx, csv, xml")
        
        # Check if converted file exists
        file_path = os.path.join(CONVERTED_DIR, f"{file_id}.{format}")
        
        # If file doesn't exist, try to generate it from JSON
        if not os.path.exists(file_path):
            json_path = os.path.join(CONVERTED_DIR, f"{file_id}.json")
            if not os.path.exists(json_path):
                raise HTTPException(status_code=404, detail="Converted data not found. Please convert the file first.")
            
            # Load JSON data and convert to requested format
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Convert to pandas DataFrame
                if isinstance(json_data, dict) and 'rows' in json_data:
                    df = pd.DataFrame(json_data['rows'])
                else:
                    df = pd.DataFrame(json_data)
                
                # Validate data before generating file
                from validation_utils import validate_financial_data
                
                # Convert DataFrame to list of dictionaries for validation
                data_list = df.to_dict('records')
                validation_result = validate_financial_data(data_list, strict=False)
                
                if validation_result['error_count'] > 0:
                    # Log validation warnings but don't block download
                    logger.warning(f"Validation warnings for {file_id}: {validation_result['errors']}")
                
                # Generate the requested format
                if format == 'xlsx':
                    df.to_excel(file_path, index=False, engine='openpyxl')
                elif format == 'csv':
                    df.to_csv(file_path, index=False, encoding='utf-8')
                elif format == 'xml':
                    create_tally_xml(df, file_path)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error generating {format} file: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to generate {format} file")
        
        # Generate dynamic filename
        try:
            from name_extractor import generate_dynamic_filename
            from file_metadata import get_file_metadata
            
            metadata = get_file_metadata(file_id)
            original_filename = metadata.get('original_filename') if metadata else None
            extracted_name = metadata.get('extracted_name') if metadata else None
            
            dynamic_filename = generate_dynamic_filename(
                file_id=file_id,
                original_filename=original_filename,
                extracted_name=extracted_name,
                file_format=format
            )
        except Exception as e:
            logger.warning(f"Could not generate dynamic filename: {e}")
            dynamic_filename = f"{file_id}.{format}"
        
        # Stream the file
        def iterfile():
            with open(file_path, mode="rb") as file_like:
                yield from file_like
        
        return StreamingResponse(
            iterfile(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if format == 'xlsx'
                      else "text/csv" if format == 'csv'
                      else "application/xml",
            headers={
                "Content-Disposition": f"attachment; filename={dynamic_filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/filename/{file_id}")
async def get_dynamic_filename(file_id: str, format: str = 'xlsx', language: str = 'en'):
    """Get dynamic filename for a file"""
    try:
        from name_extractor import generate_dynamic_filename
        from file_metadata import get_file_metadata
        
        metadata = get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File metadata not found")
            
        original_filename = metadata.get('original_filename')
        extracted_name = metadata.get('extracted_name')
        
        dynamic_filename = generate_dynamic_filename(
            file_id=file_id,
            original_filename=original_filename,
            extracted_name=extracted_name,
            file_format=format,
            language=language
        )
        
        return {
            "filename": dynamic_filename,
            "metadata": {
                "original_filename": original_filename,
                "extracted_name": extracted_name,
                "has_extracted_name": bool(extracted_name)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating filename: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_reconciliation_report(filename: str):
    """Download reconciliation report"""
    try:
        file_path = os.path.join(CONVERTED_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        def iterfile():
            with open(file_path, mode="rb") as file_like:
                yield from file_like
        
        return StreamingResponse(
            iterfile(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/convert-progress/{file_id}")
async def get_conversion_progress(file_id: str):
    """Get the current conversion progress for a file"""
    try:
        progress = progress_tracker.get_progress(file_id)
        return JSONResponse(content=progress)
    except Exception as e:
        logger.error(f"Error getting progress: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get progress", "stage": "error", "progress": 0}
        )

@app.delete("/convert-progress/{file_id}")
async def clear_conversion_progress(file_id: str):
    """Clear conversion progress data for a file"""
    try:
        progress_tracker.clear_progress(file_id)
        return JSONResponse(content={"message": "Progress data cleared"})
    except Exception as e:
        logger.error(f"Error clearing progress: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to clear progress"}
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
    