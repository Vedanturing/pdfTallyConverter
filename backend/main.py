from fastapi import FastAPI, UploadFile, HTTPException, File, Request, Query, Body, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
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
from typing import List, Dict, Optional, Any
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Tally Converter API",
    description="API for converting PDF files",
    version="1.0.0"
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
app.add_middleware(GZipMiddleware, minimum_size=1000)

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

# Initialize process pool for CPU-intensive tasks
process_pool = ProcessPoolExecutor()

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

async def save_upload_file(upload_file: UploadFile) -> tuple[str, str]:
    """Save uploaded file and return the file path"""
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(upload_file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            while chunk := await upload_file.read(CHUNK_SIZE):
                await out_file.write(chunk)
        
        return file_path, file_id
    except Exception as e:
        # Clean up the file if there's an error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

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
                                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
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
    """Convert a file and return the extracted data"""
    try:
        # Extract data from file
        if file_path.lower().endswith('.pdf'):
            data = extract_tables_from_pdf(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            data = process_image_ocr(file_path)
        else:
            raise ConversionError(f"Unsupported file format: {file_path}")

        # Get headers
        headers = data.columns.tolist()

        # Convert DataFrame to rows while preserving numeric types
        rows = []
        for _, row in data.iterrows():
            row_dict = {}
            for col in headers:
                value = row[col]
                # Convert numpy types to Python types
                if pd.isna(value):
                    row_dict[col] = None
                elif isinstance(value, (np.integer, np.floating)):
                    row_dict[col] = float(value) if isinstance(value, np.floating) else int(value)
                else:
                    row_dict[col] = str(value)
            rows.append(row_dict)

        # Create output directory
        os.makedirs(CONVERTED_DIR, exist_ok=True)

        # Convert to requested formats
        output_files = {}
        df = pd.DataFrame(rows)
        
        for format in output_formats:
            output_path = os.path.join(CONVERTED_DIR, f"{file_id}.{format}")
            if format == "xlsx":
                df.to_excel(output_path, index=False)
            elif format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "xml":
                create_tally_xml(df, output_path)
            output_files[format] = output_path

        return {
            "status": "success",
            "message": "File converted successfully",
            "file_id": file_id,
            "headers": headers,
            "rows": rows,
            "converted_files": {
                format: f"/download/{os.path.basename(path)}"
                for format, path in output_files.items()
            }
        }

    except Exception as e:
        logger.error(f"Error converting file: {str(e)}")
        logger.error(traceback.format_exc())
        raise ConversionError(str(e))

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    password: str = Form(None)
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
        file_extension = os.path.splitext(file.filename)[1].lower()
        
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
async def convert_uploaded_file(file_id: str):
    """Convert a file and return the extracted data"""
    try:
        logger.info(f"Starting conversion for file_id: {file_id}")
        
        # Find the original file
        original_file = None
        upload_dir_contents = os.listdir(UPLOAD_DIR)
        logger.info(f"Files in upload directory: {upload_dir_contents}")
        
        for filename in upload_dir_contents:
            if filename.startswith(file_id):
                original_file = os.path.join(UPLOAD_DIR, filename)
                logger.info(f"Found original file: {original_file}")
                break
        
        if not original_file:
            logger.error(f"Original file not found for file_id: {file_id}")
            raise HTTPException(status_code=404, detail="Original file not found")

        # Extract data from the file
        try:
            if original_file.lower().endswith('.pdf'):
                logger.info("Processing PDF file")
                df = extract_tables_from_pdf(original_file)
            elif original_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                logger.info("Processing image file")
                df = process_image_ocr(original_file)
            else:
                raise ValueError(f"Unsupported file format: {original_file}")

            if df.empty:
                raise ValueError("No data could be extracted from the file")

            # Convert DataFrame to dictionary format while preserving numeric types
            headers = df.columns.tolist()
            rows = []
            for _, row in df.iterrows():
                row_dict = {}
                for col in headers:
                    value = row[col]
                    # Convert numpy types to Python types
                    if pd.isna(value):
                        row_dict[col] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        row_dict[col] = float(value) if isinstance(value, np.floating) else int(value)
                    elif isinstance(value, pd.Timestamp):
                        row_dict[col] = value.strftime('%Y-%m-%d')
                    else:
                        row_dict[col] = str(value)
                rows.append(row_dict)

            # Create output directory
            os.makedirs(CONVERTED_DIR, exist_ok=True)

            # Save as JSON for future use
            json_data = {
                "headers": headers,
                "rows": rows
            }
            json_path = os.path.join(CONVERTED_DIR, f"{file_id}.json")
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
            
            # Also save as Excel for compatibility
            excel_path = os.path.join(CONVERTED_DIR, f"{file_id}.xlsx")
            df.to_excel(excel_path, index=False, engine='openpyxl')
            
            return JSONResponse({
                "status": "success",
                "message": "File converted successfully",
                "file_id": file_id,
                "headers": headers,
                "rows": rows,
                "files": {
                    "json": f"/download/{file_id}.json",
                    "excel": f"/download/{file_id}.xlsx"
                }
            })

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in convert_uploaded_file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error converting file: {str(e)}"
        )

@app.get("/api/convert/{file_id}/{format}")
async def convert_to_format(file_id: str, format: str):
    """Convert a file to a specific format"""
    try:
        logger.info(f"Starting format conversion for file_id: {file_id} to {format}")
        
        # Check if we have already converted data
        converted_file = os.path.join(CONVERTED_DIR, f"{file_id}.json")
        if not os.path.exists(converted_file):
            # If not, we need to convert first
            data = await convert_uploaded_file(file_id)
        else:
            # Use existing converted data
            with open(converted_file, 'r') as f:
                data = json.load(f)
        
        # Create DataFrame from the data
        if 'data' in data and 'rows' in data['data']:
            df = pd.DataFrame(data['data']['rows'])
        elif 'rows' in data:
            df = pd.DataFrame(data['rows'])
        else:
            raise ValueError("Invalid data format")
        
        # Create output filename
        output_filename = f"{file_id}.{format}"
        output_path = os.path.join(CONVERTED_DIR, output_filename)
        
        # Ensure the converted directory exists
        os.makedirs(CONVERTED_DIR, exist_ok=True)
        
        try:
            # Convert to the requested format
            if format == "xlsx":
                df.to_excel(output_path, index=False, engine='openpyxl')
                media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif format == "csv":
                df.to_csv(output_path, index=False)
                media_type = "text/csv"
            elif format == "xml":
                create_tally_xml(df, output_path)
                media_type = "application/xml"
            else:
                raise HTTPException(status_code=400, detail="Unsupported format")

            logger.info(f"Successfully converted to {format}")
            
            if not os.path.exists(output_path):
                logger.error(f"Output file was not created: {output_path}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create output file"
                )

            # Return the converted file with proper headers
            headers = {
                'Content-Disposition': f'attachment; filename="converted-file.{format}"',
                'Access-Control-Expose-Headers': 'Content-Disposition'
            }
            
            return FileResponse(
                path=output_path,
                media_type=media_type,
                filename=f"converted-file.{format}",
                headers=headers
            )

        except Exception as e:
            logger.error(f"Error converting to format: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error converting to format: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.post("/convert")
async def convert_new_file(
    file: UploadFile = File(...),
    output_formats: List[str] = ["xlsx", "csv", "xml"]
):
    """Convert a new file upload"""
    try:
        # Save uploaded file
        file_path, file_id = await save_upload_file(file)
        
        # Convert the file
        result = convert_file(file_path, file_id, output_formats)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return JSONResponse(result)
        
    except ConversionError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Internal server error: {str(e)}"}
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a file from the corrected directory"""
    try:
        file_path = Path("corrected") / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=get_media_type(filename)
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def get_media_type(filename: str) -> str:
    """Get the media type based on file extension"""
    ext = filename.split(".")[-1].lower()
    media_types = {
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "xml": "application/xml"
    }
    return media_types.get(ext, "application/octet-stream")

def ensure_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def save_as_xlsx(data: TableData, filepath: str):
    # Convert to pandas DataFrame
    rows_data = []
    for row in data.rows:
        row_dict = {header: row.cells[header].value for header in data.headers}
        rows_data.append(row_dict)
    
    df = pd.DataFrame(rows_data, columns=data.headers)
    df.to_excel(filepath, index=False)

def save_as_csv(data: TableData, filepath: str):
    # Convert to pandas DataFrame
    rows_data = []
    for row in data.rows:
        row_dict = {header: row.cells[header].value for header in data.headers}
        rows_data.append(row_dict)
    
    df = pd.DataFrame(rows_data, columns=data.headers)
    df.to_csv(filepath, index=False)

def save_as_xml(data: TableData, filepath: str):
    root = ET.Element("ENVELOPE")
    header = ET.SubElement(root, "HEADER")
    ET.SubElement(header, "VERSION").text = "1"
    ET.SubElement(header, "TALLYREQUEST").text = "Import Data"
    
    body = ET.SubElement(root, "BODY")
    importdata = ET.SubElement(body, "IMPORTDATA")
    requestdesc = ET.SubElement(importdata, "REQUESTDESC")
    reportname = ET.SubElement(requestdesc, "REPORTNAME").text = "Custom"
    
    requestdata = ET.SubElement(importdata, "REQUESTDATA")
    
    # Convert rows to XML structure
    for row in data.rows:
        tallymessage = ET.SubElement(requestdata, "TALLYMESSAGE")
        for header in data.headers:
            ET.SubElement(tallymessage, str(header)).text = str(row.cells[header].value)
    
    # Save XML file
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding='utf-8', xml_declaration=True)

def log_changes(file_id: str, edit_history: List[EditHistory]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    ensure_directory(log_dir)
    
    log_file = os.path.join(log_dir, f"{file_id}_{timestamp}_changes.json")
    with open(log_file, "w") as f:
        json.dump([edit.dict() for edit in edit_history], f, indent=2)

@app.post("/api/save-edits")
async def save_edits(payload: SavePayload):
    try:
        # Ensure the corrected directory exists
        corrected_dir = "corrected"
        ensure_directory(corrected_dir)
        
        # Save in different formats
        base_path = os.path.join(corrected_dir, f"{payload.fileId}_corrected")
        save_as_xlsx(payload.modifiedData, f"{base_path}.xlsx")
        save_as_csv(payload.modifiedData, f"{base_path}.csv")
        save_as_xml(payload.modifiedData, f"{base_path}.xml")
        
        # Log the changes
        log_changes(payload.fileId, payload.editHistory)
        
        return {
            "success": True,
            "downloads": {
                "xlsx": f"/api/download/{payload.fileId}/xlsx",
                "csv": f"/api/download/{payload.fileId}/csv",
                "xml": f"/api/download/{payload.fileId}/xml"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{file_id}/{format}")
async def download_converted_file(file_id: str, format: str):
    try:
        # Check if the file exists in converted directory
        converted_dir = "converted"
        filename = f"{file_id}.{format}"
        file_path = os.path.join(converted_dir, filename)
        
        if not os.path.exists(file_path):
            # Try to generate the file from the JSON data
            json_path = os.path.join(converted_dir, f"{file_id}.json")
            if not os.path.exists(json_path):
                raise HTTPException(status_code=404, detail="File not found")
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            
            if format == 'xlsx':
                output_path = os.path.join(converted_dir, f"{file_id}.xlsx")
                df.to_excel(output_path, index=False, engine='openpyxl')
                file_path = output_path
            elif format == 'csv':
                output_path = os.path.join(converted_dir, f"{file_id}.csv")
                df.to_csv(output_path, index=False)
                file_path = output_path
            elif format == 'xml':
                output_path = os.path.join(converted_dir, f"{file_id}.xml")
                # Convert DataFrame to XML
                xml_data = df.to_xml(index=False)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(xml_data)
                file_path = output_path
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        return FileResponse(
            file_path,
            filename=f"converted_file.{format}",
            media_type={
                'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'csv': 'text/csv',
                'xml': 'application/xml'
            }.get(format, 'application/octet-stream')
        )
    except Exception as e:
        logging.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    """List all uploaded files"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                file_id = os.path.splitext(filename)[0]
                file_stats = os.stat(file_path)
                mime_type, _ = mimetypes.guess_type(filename)
                
                files.append({
                    "id": file_id,
                    "name": filename,
                    "type": mime_type or "application/octet-stream",
                    "size": file_stats.st_size,
                    "uploadedAt": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "url": f"/files/{filename}"
                })
        
        return {
            "status": "success",
            "files": sorted(files, key=lambda x: x["uploadedAt"], reverse=True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{filename}")
async def serve_file(filename: str):
    """Serve an uploaded file"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    mime_type, _ = mimetypes.guess_type(filename)
    return FileResponse(
        file_path,
        media_type=mime_type or "application/octet-stream",
        filename=filename
    )

def validate_table(data: List[List[str]]) -> List[Dict]:
    """Validate table data and return validation results"""
    validation_results = []
    
    for row_idx, row in enumerate(data):
        for col_idx, cell in enumerate(row):
            # Skip header row
            if row_idx == 0:
                continue
                
            # Validate empty mandatory fields
            if not cell.strip() and col_idx in [0, 1, 2]:  # Assuming first 3 columns are mandatory
                validation_results.append({
                    "row": row_idx,
                    "column": col_idx,
                    "type": "error",
                    "severity": "critical",
                    "message": "Mandatory field cannot be empty"
                })
            
            # Validate numeric values
            if col_idx in [3, 4]:  # Assuming columns 4 and 5 should be numeric
                try:
                    float(cell.replace(',', '').strip())
                except ValueError:
                    validation_results.append({
                        "row": row_idx,
                        "column": col_idx,
                        "type": "error",
                        "severity": "critical",
                        "message": "Value must be numeric"
                    })
            
            # Validate date format
            if col_idx == 2:  # Assuming column 3 is date
                try:
                    datetime.strptime(cell.strip(), '%Y-%m-%d')
                except ValueError:
                    validation_results.append({
                        "row": row_idx,
                        "column": col_idx,
                        "type": "error",
                        "severity": "warning",
                        "message": "Invalid date format (should be YYYY-MM-DD)"
                    })
            
            # Check for common OCR errors
            if any(char in cell for char in ['O0', 'l1', 'S5']):
                validation_results.append({
                    "row": row_idx,
                    "column": col_idx,
                    "type": "warning",
                    "severity": "info",
                    "message": "Possible OCR confusion (O/0, l/1, S/5)"
                })
    
    return validation_results

@app.post("/validate/{file_id}")
async def validate_data(file_id: str, request: Request):
    """Validate converted data for a specific file"""
    try:
        # Get the request body
        data = await request.json()
        
        if not data:
            raise HTTPException(status_code=400, detail="No data provided for validation")

        # Get the file path
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Extract data and formats from request
        table_data = data.get('data', [])
        formats = data.get('formats', [])
        password = data.get('password')

        if not table_data:
            raise HTTPException(status_code=422, detail="No table data provided")
        
        # Ensure table_data is a list
        if not isinstance(table_data, list):
            table_data = [table_data]
        
        # Convert to DataFrame for validation
        try:
            df = pd.DataFrame(table_data)
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid data format: {str(e)}"
            )
        
        # Clean up data types
        try:
            # Convert date columns if they exist
            date_columns = [col for col in df.columns if any(term in col.lower() for term in ['date', 'dt', 'time'])]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to date: {str(e)}")
            
            # Convert numeric columns if they exist
            numeric_columns = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'balance', 'total', 'price', 'quantity'])]
            for col in numeric_columns:
                try:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
            
            # Fill NaN values
            df = df.fillna('')
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Error cleaning data: {str(e)}"
            )
        
        try:
            # Convert DataFrame back to records for validation
            records = df.to_dict('records')
            
            # Perform validation checks
            validation_results = validate_table_data(records)
            financial_validation = validate_financial_data(records)
            
            # Get validation summary
            summary = get_validation_summary(records)
            
            # Save validated data
            try:
                os.makedirs("corrected", exist_ok=True)
                validated_path = os.path.join("corrected", f"{file_id}_validated.json")
                with open(validated_path, 'w') as f:
                    json.dump({
                        'data': records,
                        'formats': formats,
                        'validation': {
                            'results': validation_results,
                            'financial': financial_validation,
                            'summary': summary
                        }
                    }, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving validated data: {str(e)}")
                # Don't fail the validation if saving fails
                pass
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "Validation completed successfully",
                    "validationResults": {
                        "results": validation_results,
                        "financial": financial_validation,
                        "summary": summary
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=422,
                detail=f"Validation error: {str(e)}"
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during validation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.post("/export/{format}")
async def export_data(format: str, data: dict):
    """Export data to various formats"""
    try:
        if not data or not data.get('data'):
            raise HTTPException(status_code=400, detail="No data provided")

        df = pd.DataFrame(data['data'])
        output_file = f"temp_export.{format}"

        if format == 'xlsx':
            df.to_excel(output_file, index=False, engine='openpyxl')
            media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif format == 'csv':
            df.to_csv(output_file, index=False)
            media_type = 'text/csv'
        elif format == 'xml':
            # Create XML structure
            root = ET.Element("FinancialData")
            for _, row in df.iterrows():
                entry = ET.SubElement(root, "Entry")
                for col in df.columns:
                    ET.SubElement(entry, col).text = str(row[col])
            
            tree = ET.ElementTree(root)
            tree.write(output_file, encoding='utf-8', xml_declaration=True)
            media_type = 'application/xml'
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

        # Return file and clean up
        response = FileResponse(
            output_file,
            media_type=media_type,
            filename=f"financial_data.{format}"
        )

        # Clean up in background after response is sent
        background_tasks = BackgroundTasks()
        background_tasks.add_task(lambda: os.remove(output_file))
        
        return response

    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/{file_id}/{format}")
async def export_data(
    file_id: str,
    format: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    try:
        data = await request.json()
        client_name = data.get("clientName", "export")
        
        # Get the validated data
        validated_path = os.path.join("corrected", f"{file_id}_validated.json")
        if not os.path.exists(validated_path):
            raise HTTPException(status_code=404, detail="Validated data not found")
        
        with open(validated_path, "r") as f:
            entries = json.load(f)
        
        # Create export directory
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y%m%d")
        safe_client_name = "".join(c for c in client_name if c.isalnum() or c in (' ', '-', '_')).strip()
        base_filename = f"{safe_client_name}_{date_str}"
        
        if format == "xlsx":
            # Standard Excel export
            df = pd.DataFrame(entries)
            output_path = os.path.join(export_dir, f"{base_filename}.xlsx")
            df.to_excel(output_path, index=False, engine="openpyxl")
            
        elif format == "tally":
            # Tally-compatible Excel format
            df = pd.DataFrame(entries)
            # Add Tally-specific formatting
            df["Voucher Type"] = "Receipt"
            df["Narration"] = df["description"]
            output_path = os.path.join(export_dir, f"{base_filename}_tally.xlsx")
            df.to_excel(output_path, index=False, engine="openpyxl")
            
        elif format == "json":
            # JSON export
            output_path = os.path.join(export_dir, f"{base_filename}.json")
            with open(output_path, "w") as f:
                json.dump(entries, f, indent=2)
                
        elif format == "pdf":
            # Generate PDF summary
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
            
            output_path = os.path.join(export_dir, f"{base_filename}_summary.pdf")
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            
            # Create the PDF content
            styles = getSampleStyleSheet()
            elements = []
            
            # Add title
            elements.append(Paragraph(f"Financial Summary - {client_name}", styles["Title"]))
            elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
            
            # Create table
            df = pd.DataFrame(entries)
            table_data = [df.columns.tolist()] + df.values.tolist()
            t = Table(table_data)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 14),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 12),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t)
            
            # Build PDF
            doc.build(elements)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
        # Log the export
        log_path = os.path.join("exports", "audit_log.json")
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "client_name": client_name,
            "file_id": file_id,
            "format": format,
            "filename": os.path.basename(output_path)
        }
        
        try:
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    log = json.load(f)
            else:
                log = []
            
            log.append(log_entry)
            
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            logging.error(f"Error writing to audit log: {str(e)}")
        
        # Clean up old exports in the background
        background_tasks.add_task(cleanup_old_exports, export_dir)
        
        return FileResponse(
            output_path,
            filename=os.path.basename(output_path),
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logging.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_old_exports(export_dir: str, max_age_days: int = 7):
    """Clean up export files older than max_age_days."""
    try:
        current_time = datetime.now()
        for file in os.listdir(export_dir):
            if file == "audit_log.json":
                continue
                
            file_path = os.path.join(export_dir, file)
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if (current_time - file_modified).days > max_age_days:
                os.remove(file_path)
                logging.info(f"Cleaned up old export: {file}")
    except Exception as e:
        logging.error(f"Error cleaning up exports: {str(e)}")

@app.post("/bank-statement/upload")
async def upload_bank_statement(file: UploadFile = File(...)):
    """Upload and process a bank statement"""
    try:
        file_path, file_id = await save_upload_file(file)
        
        # Process the bank statement
        transactions = bank_matcher.load_bank_statement(file_path)
        
        return {
            "success": True,
            "message": "Bank statement processed successfully",
            "transaction_count": len(transactions)
        }
    except Exception as e:
        logger.error(f"Error processing bank statement: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing bank statement: {str(e)}"
        )

@app.post("/invoice-data/upload")
async def upload_invoice_data(file: UploadFile = File(...)):
    """Upload and process invoice data"""
    try:
        file_path, file_id = await save_upload_file(file)
        
        # Process the invoice data based on file type
        if file_path.lower().endswith('.pdf'):
            df = process_bank_statement(file_path)  # Reuse bank statement parser
        else:
            df = pd.read_csv(file_path)
        
        # Convert DataFrame to list of dictionaries
        invoice_data = df.to_dict('records')
        
        # Load into bank matcher
        transactions = bank_matcher.load_invoice_data(invoice_data)
        
        return {
            "success": True,
            "message": "Invoice data processed successfully",
            "transaction_count": len(transactions)
        }
    except Exception as e:
        logger.error(f"Error processing invoice data: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing invoice data: {str(e)}"
        )

@app.post("/bank-matcher/match")
async def match_transactions(
    date_tolerance_days: int = Query(5, description="Number of days to consider for matching dates"),
    amount_tolerance: float = Query(0.01, description="Amount difference tolerance for matching")
):
    """Find matches between bank transactions and invoices"""
    try:
        matches = bank_matcher.find_matches(
            date_tolerance_days=date_tolerance_days,
            amount_tolerance=amount_tolerance
        )
        
        unmatched = bank_matcher.get_unmatched_transactions()
        
        return {
            "success": True,
            "matches": matches,
            "unmatched": unmatched,
            "match_count": len(matches)
        }
    except Exception as e:
        logger.error(f"Error matching transactions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error matching transactions: {str(e)}"
        )

@app.post("/bank-matcher/update-status")
async def update_match_status(
    match_id: str = Body(..., embed=True),
    status: str = Body(..., embed=True)
):
    """Update the status of a match"""
    try:
        success = bank_matcher.update_match_status(match_id, status)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Match with ID {match_id} not found"
            )
        
        return {
            "success": True,
            "message": f"Match status updated to {status}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating match status: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error updating match status: {str(e)}"
        )

@app.post("/gst/add-invoice")
async def add_gst_invoice(invoice_data: Dict[str, Any]):
    """Add an invoice for GST processing"""
    try:
        invoice = gst_helper.add_invoice(invoice_data)
        return {
            "success": True,
            "message": "Invoice added successfully",
            "invoice": {
                "invoice_no": invoice.invoice_no,
                "total_amount": invoice.total_amount,
                "gst_details": {
                    "taxable_value": invoice.taxable_value,
                    "igst": invoice.igst,
                    "cgst": invoice.cgst,
                    "sgst": invoice.sgst
                }
            }
        }
    except Exception as e:
        logger.error(f"Error adding GST invoice: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error adding GST invoice: {str(e)}"
        )

@app.post("/gst/generate-gstr1")
async def generate_gstr1(period: str = Body(..., embed=True)):
    """Generate GSTR-1 format JSON"""
    try:
        gstr1_data = gst_helper.generate_gstr1_json(period)
        
        # Save to file
        file_path = os.path.join(EXPORT_DIR, f"GSTR1_{period}.json")
        with open(file_path, 'w') as f:
            json.dump(gstr1_data, f, indent=2)
        
        return FileResponse(
            file_path,
            media_type='application/json',
            filename=f"GSTR1_{period}.json"
        )
    except Exception as e:
        logger.error(f"Error generating GSTR-1: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error generating GSTR-1: {str(e)}"
        )

@app.post("/gst/generate-gstr3b")
async def generate_gstr3b(period: str = Body(..., embed=True)):
    """Generate GSTR-3B format JSON"""
    try:
        gstr3b_data = gst_helper.generate_gstr3b_json(period)
        
        # Save to file
        file_path = os.path.join(EXPORT_DIR, f"GSTR3B_{period}.json")
        with open(file_path, 'w') as f:
            json.dump(gstr3b_data, f, indent=2)
        
        return FileResponse(
            file_path,
            media_type='application/json',
            filename=f"GSTR3B_{period}.json"
        )
    except Exception as e:
        logger.error(f"Error generating GSTR-3B: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error generating GSTR-3B: {str(e)}"
        )

@app.post("/gst/generate-excel")
async def generate_gst_excel(period: str = Body(..., embed=True)):
    """Generate GST Excel report"""
    try:
        file_path = os.path.join(EXPORT_DIR, f"GST_Report_{period}.xlsx")
        gst_helper.generate_excel_report(file_path)
        
        return FileResponse(
            file_path,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=f"GST_Report_{period}.xlsx"
        )
    except Exception as e:
        logger.error(f"Error generating GST Excel report: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error generating GST Excel report: {str(e)}"
        )

@app.post("/gst/validate-gstin")
async def validate_gstin(gstin: str = Body(..., embed=True)):
    """Validate GSTIN format"""
    try:
        is_valid = gst_helper.validate_gstin(gstin)
        return {
            "success": True,
            "is_valid": is_valid
        }
    except Exception as e:
        logger.error(f"Error validating GSTIN: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error validating GSTIN: {str(e)}"
        )

@app.get("/file/{file_id}")
async def get_file(file_id: str, convert: bool = Query(False)):
    """Get file by ID and optionally convert it to table format"""
    try:
        logger.info(f"Processing file request: file_id={file_id}, convert={convert}")
        
        # Check all possible file extensions
        file_found = False
        file_path = None
        file_ext = None
        
        for ext in ['.pdf', '.png', '.jpg', '.jpeg']:
            test_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
            logger.info(f"Checking file path: {test_path}")
            if os.path.exists(test_path):
                file_path = test_path
                file_ext = ext
                file_found = True
                logger.info(f"File found: {file_path}")
                break
        
        if not file_found:
            logger.error(f"File not found for ID: {file_id}")
            # List available files for debugging
            try:
                available_files = os.listdir(UPLOAD_DIR)
                logger.info(f"Available files in upload directory: {available_files}")
            except Exception as e:
                logger.error(f"Could not list upload directory: {str(e)}")
            raise HTTPException(status_code=404, detail="File not found")
        
        if not convert:
            # Return raw file if conversion not requested
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            return FileResponse(
                file_path,
                media_type=mime_type,
                filename=os.path.basename(file_path)
            )
        else:
            # Convert file to table format
            logger.info(f"Starting conversion for file: {file_path}")
            try:
                df = None
                if file_ext.lower() == '.pdf':
                    logger.info("Processing PDF file")
                    df = extract_tables_from_pdf(file_path)
                else:
                    logger.info("Processing image file")
                    df = process_image_ocr(file_path)
                
                logger.info(f"Conversion result: df is None: {df is None}, df is empty: {df.empty if df is not None else 'N/A'}")
                
                # Convert DataFrame to response format
                if df is not None and not df.empty:
                    logger.info(f"DataFrame shape: {df.shape}")
                    logger.info(f"DataFrame columns: {list(df.columns)}")
                    
                    try:
                        rows = df.to_dict('records')
                        headers = df.columns.tolist()
                        
                        logger.info(f"Successfully converted to {len(rows)} rows with {len(headers)} columns")
                        
                        return {
                            "success": True,
                            "data": {
                                "rows": rows,
                                "headers": headers
                            }
                        }
                    except Exception as e:
                        logger.error(f"Error converting DataFrame to dict: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise HTTPException(
                            status_code=500,
                            detail=f"Error formatting conversion results: {str(e)}"
                        )
                else:
                    logger.warning("No data extracted from file or DataFrame is empty")
                    raise HTTPException(
                        status_code=422,
                        detail="No tabular data could be extracted from the file. The file may not contain tables or the format may not be supported."
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error converting file {file_id}: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(
                    status_code=500,
                    detail=f"Error converting file: {str(e)}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_file for {file_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {str(e)}"
        )

@app.get("/audit-logs")
async def get_audit_logs(
    limit: Optional[int] = None,
    action_type: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Get audit logs with optional filtering"""
    return audit_logger.get_logs(limit=limit, action_type=action_type, user_id=user_id)

@app.post("/audit-logs")
async def create_audit_log(
    action_type: str = Body(...),
    summary: str = Body(...),
    user_id: Optional[str] = Body(None),
    metadata: Optional[Dict[str, Any]] = Body(None)
):
    """Create a new audit log entry"""
    return audit_logger.log_action(
        action_type=action_type,
        summary=summary,
        user_id=user_id,
        metadata=metadata
    )

@app.get("/missed-rows/{file_id}")
async def get_missed_rows(file_id: str):
    """Get rows that failed to parse from bank statement"""
    try:
        # Get the file path
        file_path = None
        for ext in ['.pdf', '.png', '.jpg', '.jpeg']:
            path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Process the file and get missed rows
        parser = BankStatementParser()
        _, missed_rows = parser.process_bank_statement(file_path)
        
        return {
            "success": True,
            "missed_rows": missed_rows
        }
        
    except Exception as e:
        logger.error(f"Error getting missed rows: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error getting missed rows: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
    