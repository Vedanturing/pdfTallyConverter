from fastapi import FastAPI, UploadFile, HTTPException, File, Request, Query, Body
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
import uvicorn
import mimetypes
from fastapi import BackgroundTasks
from pathlib import Path
import re
from validation_utils import validate_table_data, get_validation_summary, validate_financial_data, ValidationError
from bank_statement_parser import process_bank_statement, validate_numeric_fields
from bank_matcher import BankMatcher
from gst_helper import GSTHelper

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
    logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Request completed: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(e)}
        )

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

def extract_tables_from_pdf(file_path: str) -> pd.DataFrame:
    """Extract tables from PDF using pdfplumber with fallback to bank statement parser"""
    logger.info(f"Extracting tables from PDF: {file_path}")
    try:
        # Try general table extraction first
        with pdfplumber.open(file_path) as pdf:
            all_tables = []
            for page_num, page in enumerate(pdf.pages, 1):
                # Try with default settings first
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
                else:
                    # If no tables found, try with custom settings
                    tables = page.find_tables({
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "text_tolerance": 3,
                        "text_x_tolerance": 3,
                        "intersection_tolerance": 3,
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                    }).extract()
                    if tables:
                        all_tables.extend(tables)
            
            if all_tables:
                # Use the first table's headers
                headers = all_tables[0][0]
                # Combine all rows from all tables
                rows = []
                for table in all_tables:
                    rows.extend(table[1:])  # Skip header row for subsequent tables
                
                df = pd.DataFrame(rows, columns=headers)
                df = df.fillna('')  # Replace NaN with empty string
                
                # Try to detect if this is tabular data
                if len(df.columns) >= 3 and len(df) > 0:
                    # Standardize column names
                    column_mapping = {
                        col: col.lower().replace(' ', '_') 
                        for col in df.columns
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
                        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
                        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                    
                    # Clean up amount columns
                    if 'amount' in df.columns:
                        df['amount'] = df['amount'].apply(lambda x: str(x).replace('₹', '').replace(',', '').strip())
                        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
                    
                    # Clean up balance column
                    if 'balance' in df.columns:
                        df['balance'] = df['balance'].apply(lambda x: str(x).replace('₹', '').replace(',', '').strip())
                        df['balance'] = pd.to_numeric(df['balance'], errors='coerce').fillna(0)
                    
                    return df
        
        # If general table extraction fails or doesn't look like tabular data,
        # try processing as bank statement
        logger.info("General table extraction failed or invalid, trying bank statement parser...")
        df = process_bank_statement(file_path)
        
        if not df.empty:
            # Validate numeric fields
            validation_results = validate_numeric_fields(df)
            if validation_results['errors']:
                logger.warning(f"Found {len(validation_results['errors'])} numeric field errors")
            if validation_results['warnings']:
                logger.info(f"Found {len(validation_results['warnings'])} warnings")
            
            # Convert DataFrame to expected format
            df = df.fillna('')  # Replace NaN with empty string
            
            # Ensure required columns exist
            required_columns = ['date', 'narration', 'amount', 'balance']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            
            # Format numeric columns
            for col in ['amount', 'balance']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: '{:.2f}'.format(float(x)) if x != '' else '')
            
            # Format date column
            if 'date' in df.columns and df['date'].any():
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            return df
        
        # If both methods fail, return empty DataFrame
        logger.warning("Both table extraction methods failed")
        return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in table extraction: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def process_image_ocr(file_path: str) -> pd.DataFrame:
    """Process image using OCR to extract tabular data"""
    try:
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            raise ConversionError("Failed to read image file")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # OCR
        text = pytesseract.image_to_string(gray)
        
        # Convert OCR text to DataFrame
        # This is a simple implementation - you might need to enhance it based on your needs
        lines = text.strip().split('\n')
        if not lines:
            raise ConversionError("No text detected in image")
            
        # Assume first line contains headers
        headers = lines[0].split()
        data = [line.split() for line in lines[1:]]
        
        return pd.DataFrame(data, columns=headers)
    except Exception as e:
        raise ConversionError(f"Failed to process image: {str(e)}")

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
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and return its ID"""
    # Validate file size (e.g., 100MB limit)
    if await file.read(1) == b'':
        raise HTTPException(status_code=400, detail="Empty file")
    await file.seek(0)  # Reset file pointer
    
    # Validate file type
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.pdf', '.png', '.jpg', '.jpeg']:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    try:
        file_path, file_id = await save_upload_file(file)
        return JSONResponse({
            "status": "success",
            "message": "File uploaded successfully",
            "file_id": file_id
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
                "success": True,
                "message": "File converted successfully",
                "data": json_data
            })

        except Exception as e:
            logger.error(f"Error converting file: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=422,
                detail=f"Error converting file: {str(e)}"
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

            # Return the converted file
            return FileResponse(
                output_path,
                media_type=media_type,
                filename=f"converted-file.{format}"
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

@app.post("/validate-data")
async def validate_data(request: Request):
    try:
        data = await request.json()
        validation_results = validate_table_data(data)
        summary = get_validation_summary(validation_results)
        
        return {
            "validation_results": validation_results,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in validate_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-edits")
async def save_edits(
    request: Request,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Save edited data and export to different formats"""
    try:
        data = await request.json()
        file_id = data.get("fileId")
        original_data = data.get("originalData")
        modified_data = data.get("modifiedData")
        
        if not all([file_id, original_data, modified_data]):
            raise HTTPException(status_code=400, message="Missing required data")
            
        # Create corrected directory if it doesn't exist
        corrected_dir = Path("corrected")
        corrected_dir.mkdir(exist_ok=True)
        
        # Convert data to pandas DataFrame
        df = pd.DataFrame(modified_data)
        
        # Save as Excel
        excel_path = corrected_dir / f"{file_id}_corrected.xlsx"
        df.to_excel(excel_path, index=False)
        
        # Save as CSV
        csv_path = corrected_dir / f"{file_id}_corrected.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as XML
        xml_path = corrected_dir / f"{file_id}_corrected.xml"
        df.to_xml(xml_path, index=False)
        
        # Log changes
        changes = []
        for i, (orig, mod) in enumerate(zip(original_data, modified_data)):
            for key in orig:
                if orig[key] != mod[key]:
                    changes.append({
                        "row": i,
                        "column": key,
                        "original": orig[key],
                        "modified": mod[key]
                    })
        
        # Save changes log
        log_path = corrected_dir / f"{file_id}_changes.json"
        with open(log_path, "w") as f:
            json.dump(changes, f, indent=2)
        
        return {
            "message": "Files saved successfully",
            "files": {
                "excel": str(excel_path),
                "csv": str(csv_path),
                "xml": str(xml_path)
            }
        }
        
    except Exception as e:
        logger.error(f"Error saving edits: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/exports")
async def list_exports():
    """List all exported files"""
    try:
        files = []
        for filename in os.listdir(CORRECTED_DIR):
            if not filename.startswith('corrected_'):
                continue
                
            base_name = os.path.splitext(filename)[0]
            base_path = os.path.join(CORRECTED_DIR, base_name)
            
            # Get stats from any of the exported files (using .json as reference)
            json_path = f"{base_path}.json"
            if not os.path.exists(json_path):
                continue
                
            file_stats = os.stat(json_path)
            
            # Check which formats are available
            formats = {}
            for ext in ['json', 'xlsx', 'csv', 'xml']:
                file_path = f"{base_path}.{ext}"
                if os.path.exists(file_path):
                    formats[ext.replace('xlsx', 'excel')] = f"/exports/{base_name}.{ext}"
            
            files.append({
                "id": base_name,
                "name": base_name.replace('corrected_', ''),
                "type": "application/json",
                "size": file_stats.st_size,
                "createdAt": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "formats": formats
            })
        
        return {
            "status": "success",
            "files": sorted(files, key=lambda x: x["createdAt"], reverse=True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/exports/{filename}")
async def serve_export(filename: str):
    """Serve an exported file"""
    file_path = os.path.join(CORRECTED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    mime_type, _ = mimetypes.guess_type(filename)
    return FileResponse(
        file_path,
        media_type=mime_type or "application/octet-stream",
        filename=filename
    )

@app.get("/file/{file_id}")
async def get_file(file_id: str):
    """Get file by ID"""
    # Check all possible file extensions
    for ext in ['.pdf', '.png', '.jpg', '.jpeg']:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
        if os.path.exists(file_path):
            # Get the MIME type based on file extension
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            return FileResponse(
                file_path,
                media_type=mime_type,
                filename=os.path.basename(file_path)
            )
    
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/convert")
async def convert_to_format(fileId: str, format: str):
    """Convert a file to the specified format and return it"""
    try:
        # Get the original file path
        file_path = None
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(fileId):
                file_path = os.path.join(UPLOAD_DIR, filename)
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")

        # Create output filename
        output_filename = f"{fileId}.{format}"
        output_path = os.path.join(CONVERTED_DIR, output_filename)

        # Convert the file
        if format == "xlsx":
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            df.to_excel(output_path, index=False)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif format == "csv":
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            df.to_csv(output_path, index=False)
            media_type = "text/csv"
        elif format == "xml":
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            create_tally_xml(df, output_path)
            media_type = "application/xml"
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

        return FileResponse(
            output_path,
            media_type=media_type,
            filename=f"converted-file.{format}"
        )

    except Exception as e:
        logger.error(f"Error converting file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-data/{file_id}")
async def get_data(file_id: str):
    """Get data from a file, converting it if necessary"""
    try:
        # Check if converted data exists
        converted_file = os.path.join(CONVERTED_DIR, f"{file_id}.json")
        
        # If converted data doesn't exist, convert it first
        if not os.path.exists(converted_file):
            # Find the original file
            original_file = None
            for filename in os.listdir(UPLOAD_DIR):
                if filename.startswith(file_id):
                    original_file = os.path.join(UPLOAD_DIR, filename)
                    break
            
            if not original_file:
                raise HTTPException(status_code=404, detail="Original file not found")

            # Convert the file based on its type
            try:
                logger.info(f"Converting file: {original_file}")
                if original_file.lower().endswith('.pdf'):
                    # Try extract_tables_from_pdf first
                    try:
                        df = extract_tables_from_pdf(original_file)
                    except Exception as e:
                        logger.warning(f"Failed to extract tables using extract_tables_from_pdf: {str(e)}")
                        logger.info("Attempting to process as bank statement...")
                        # If that fails, try process_bank_statement as fallback
                        df = process_bank_statement(original_file)
                elif original_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    df = process_image_ocr(original_file)
                else:
                    raise ValueError(f"Unsupported file format: {original_file}")

                if df.empty:
                    raise ValueError("No data could be extracted from the file")

                # Clean up the DataFrame
                # Convert date columns to string format
                date_columns = df.select_dtypes(include=['datetime64']).columns
                for col in date_columns:
                    df[col] = df[col].dt.strftime('%Y-%m-%d')

                # Convert numeric columns to float
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_columns:
                    df[col] = df[col].astype(float)

                # Convert all other columns to string
                object_columns = df.select_dtypes(include=['object']).columns
                for col in object_columns:
                    df[col] = df[col].fillna('').astype(str)

                # Convert DataFrame to dictionary format
                data = {
                    'headers': df.columns.tolist(),
                    'rows': df.to_dict('records')
                }
                
                # Save the converted data
                os.makedirs(CONVERTED_DIR, exist_ok=True)
                with open(converted_file, 'w') as f:
                    json.dump(data, f)
                
                logger.info(f"Successfully converted file. Data shape: {df.shape}")
                return data

            except Exception as e:
                logger.error(f"Error converting file: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(
                    status_code=422,
                    detail=f"Error converting file: {str(e)}"
                )
        
        # Read the existing converted data
        try:
            with open(converted_file, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error reading converted data: {str(e)}")
            # If the JSON file is corrupted, delete it and raise an error
            os.remove(converted_file)
            raise HTTPException(
                status_code=500,
                detail="Corrupted conversion data. Please try converting again."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/validate/{file_id}")
async def validate_data(file_id: str, request: Request):
    """Validate converted data for a specific file"""
    try:
        # Get the request body
        data = await request.json()
        
        if not data:
            raise HTTPException(status_code=400, detail="No data provided for validation")
        
        # Convert the data to a pandas DataFrame for validation
        try:
            # Try to get data from the request body
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid data format: {str(e)}"
            )
        
        # Ensure required columns exist
        required_columns = ['date', 'narration', 'amount', 'balance']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Clean up data types
        try:
            # Convert date column
            df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
            
            # Convert numeric columns
            for col in ['amount', 'balance']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce')
            
            # Fill NaN values
            df = df.fillna('')
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Error cleaning data: {str(e)}"
            )
        
        # Perform validation checks
        validation_results = validate_numeric_fields(df)
        financial_validation = validate_financial_data(df)
        
        # Get validation summary
        summary = get_validation_summary(validation_results, financial_validation)
        
        # Save validated data
        try:
            os.makedirs("corrected", exist_ok=True)
            validated_path = os.path.join("corrected", f"{file_id}_validated.json")
            df.to_json(validated_path, orient='records', date_format='iso')
        except Exception as e:
            logger.error(f"Error saving validated data: {str(e)}")
            # Don't fail the validation if saving fails
            pass
        
        return {
            "status": "success",
            "validation_results": validation_results,
            "financial_validation": financial_validation,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": str(e)}
        )
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
    