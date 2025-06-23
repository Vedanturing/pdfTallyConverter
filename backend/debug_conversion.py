#!/usr/bin/env python3
"""
Debug script to test file conversion functionality
"""
import os
import sys
import logging
import traceback
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports"""
    logger.info("Testing imports...")
    
    imports_to_test = [
        ('pdfplumber', 'pdfplumber'),
        ('PyMuPDF', 'fitz'),
        ('pytesseract', 'pytesseract'),
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('cv2', 'cv2'),
        ('PIL', 'Image'),
        ('PyPDF2', 'PdfReader'),
    ]
    
    failed_imports = []
    
    for package_name, import_name in imports_to_test:
        try:
            if import_name == 'pd':
                import pandas as pd
            elif import_name == 'np':
                import numpy as np
            elif import_name == 'Image':
                from PIL import Image
            elif import_name == 'PdfReader':
                from PyPDF2 import PdfReader
            else:
                __import__(import_name)
            logger.info(f"✓ {package_name} imported successfully")
        except ImportError as e:
            logger.error(f"✗ Failed to import {package_name}: {str(e)}")
            failed_imports.append(package_name)
        except Exception as e:
            logger.error(f"✗ Unexpected error importing {package_name}: {str(e)}")
            failed_imports.append(package_name)
    
    if failed_imports:
        logger.error(f"Failed imports: {failed_imports}")
        return False
    else:
        logger.info("All imports successful!")
        return True

def test_conversion_function():
    """Test the conversion functions"""
    logger.info("Testing conversion functions...")
    
    try:
        # Import main functions
        from main import extract_tables_from_pdf, process_image_ocr
        logger.info("✓ Successfully imported conversion functions")
        
        # Test with a sample file if it exists
        upload_dir = Path("uploads")
        if upload_dir.exists():
            pdf_files = list(upload_dir.glob("*.pdf"))
            image_files = list(upload_dir.glob("*.png")) + list(upload_dir.glob("*.jpg")) + list(upload_dir.glob("*.jpeg"))
            
            if pdf_files:
                test_file = pdf_files[0]
                logger.info(f"Testing PDF conversion with: {test_file}")
                try:
                    df = extract_tables_from_pdf(str(test_file))
                    logger.info(f"✓ PDF conversion completed. DataFrame shape: {df.shape if df is not None else 'None'}")
                except Exception as e:
                    logger.error(f"✗ PDF conversion failed: {str(e)}")
                    logger.error(traceback.format_exc())
            
            if image_files:
                test_file = image_files[0]
                logger.info(f"Testing image conversion with: {test_file}")
                try:
                    df = process_image_ocr(str(test_file))
                    logger.info(f"✓ Image conversion completed. DataFrame shape: {df.shape if df is not None else 'None'}")
                except Exception as e:
                    logger.error(f"✗ Image conversion failed: {str(e)}")
                    logger.error(traceback.format_exc())
        else:
            logger.warning("No uploads directory found, skipping file tests")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing conversion functions: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_file_access():
    """Test file access and permissions"""
    logger.info("Testing file access...")
    
    # Test upload directory
    upload_dir = Path("uploads")
    if upload_dir.exists():
        logger.info(f"✓ Upload directory exists: {upload_dir.absolute()}")
        
        # List files
        files = list(upload_dir.iterdir())
        logger.info(f"Found {len(files)} files in upload directory")
        
        for file_path in files[:5]:  # Check first 5 files
            if file_path.is_file():
                try:
                    # Test read access
                    with open(file_path, 'rb') as f:
                        f.read(1024)  # Read first 1KB
                    logger.info(f"✓ Can read file: {file_path.name}")
                except Exception as e:
                    logger.error(f"✗ Cannot read file {file_path.name}: {str(e)}")
    else:
        logger.warning("Upload directory does not exist")
    
    return True

def main():
    """Run all tests"""
    logger.info("Starting diagnostic tests...")
    
    results = {
        'imports': test_imports(),
        'file_access': test_file_access(),
        'conversion': test_conversion_function(),
    }
    
    logger.info("Diagnostic test results:")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name}: {status}")
    
    if all(results.values()):
        logger.info("All tests passed! The conversion system should be working.")
    else:
        logger.error("Some tests failed. Check the logs above for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 