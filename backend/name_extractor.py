#!/usr/bin/env python3
"""
Name Extractor Module

Extracts bearer names from bank statements using OCR and generates
dynamic filenames based on extracted names, original filenames, and timestamps.
"""

import os
import re
import cv2
import numpy as np
import fitz  # PyMuPDF
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from PIL import Image
import unicodedata

# Try to import OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class NameExtractor:
    """Extract bearer names from bank statements"""
    
    def __init__(self):
        self.name_patterns = [
            # Indian name patterns
            r'(?:Account\s+Holder|A/C\s+Holder|Name|Customer)[\s:]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Dear\s+(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:Name|Customer\s+Name)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # Additional patterns for common bank statement formats
            r'([A-Z][A-Z\s]+[A-Z])(?:\s+A/C|Account)',  # ALL CAPS names
            r'(\b[A-Z][a-z]+\s+[A-Z][a-z]+\b)(?=.*(?:Statement|Account|Bank))',  # Common name formats
        ]
        
        # Common titles to clean from names
        self.titles_to_remove = ['MR', 'MRS', 'MS', 'DR', 'SHRI', 'SMT', 'MISS', 'MASTER']
        
        # Initialize OCR engines
        self.easy_ocr = None
        self.paddle_ocr = None
        self._init_ocr_engines()
    
    def _init_ocr_engines(self):
        """Initialize available OCR engines"""
        try:
            if EASYOCR_AVAILABLE:
                self.easy_ocr = easyocr.Reader(['en'])
                logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
            
        try:
            if PADDLEOCR_AVAILABLE:
                self.paddle_ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
                logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
    
    def extract_name_from_pdf(self, file_path: str, password: Optional[str] = None) -> Optional[str]:
        """Extract bearer name from PDF bank statement"""
        try:
            logger.info(f"Attempting to extract name from PDF: {file_path}")
            
            # Open PDF
            doc = fitz.open(file_path)
            if password:
                doc.authenticate(password)
            
            # Focus on first 2 pages as they usually contain account holder info
            max_pages = min(2, doc.page_count)
            
            for page_num in range(max_pages):
                page = doc[page_num]
                
                # First try text extraction (faster)
                text = page.get_text()
                name = self._extract_name_from_text(text)
                if name:
                    logger.info(f"Name extracted from text: {name}")
                    doc.close()
                    return name
                
                # If text extraction fails, try OCR on upper portion
                try:
                    # Get upper 40% of the page (where account holder info is typically located)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling
                    img_data = pix.tobytes("png")
                    img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Crop to upper portion
                    height, width = img_array.shape[:2]
                    upper_img = img_array[:int(height * 0.4), :]
                    
                    # Try OCR on cropped image
                    name = self._extract_name_with_ocr(upper_img)
                    if name:
                        logger.info(f"Name extracted from OCR: {name}")
                        doc.close()
                        return name
                        
                except Exception as e:
                    logger.warning(f"OCR extraction failed for page {page_num}: {e}")
                    continue
            
            doc.close()
            logger.warning("No name found in PDF")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting name from PDF: {e}")
            return None
    
    def extract_name_from_image(self, file_path: str) -> Optional[str]:
        """Extract bearer name from image bank statement"""
        try:
            logger.info(f"Attempting to extract name from image: {file_path}")
            
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                logger.error("Failed to load image")
                return None
            
            # Focus on upper portion of image
            height, width = img.shape[:2]
            upper_img = img[:int(height * 0.4), :]
            
            return self._extract_name_with_ocr(upper_img)
            
        except Exception as e:
            logger.error(f"Error extracting name from image: {e}")
            return None
    
    def _extract_name_from_text(self, text: str) -> Optional[str]:
        """Extract name from plain text using regex patterns"""
        if not text:
            return None
        
        logger.debug(f"Searching for name in text: {text[:200]}...")
        
        for pattern in self.name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                for match in matches:
                    name = self._clean_name(match)
                    if self._is_valid_name(name):
                        logger.debug(f"Found valid name with pattern '{pattern}': {name}")
                        return name
        
        return None
    
    def _extract_name_with_ocr(self, img: np.ndarray) -> Optional[str]:
        """Extract name using OCR engines"""
        # Preprocess image for better OCR
        processed_img = self._preprocess_for_ocr(img)
        
        # Try multiple OCR engines
        ocr_results = []
        
        # Try EasyOCR
        if self.easy_ocr:
            try:
                results = self.easy_ocr.readtext(processed_img)
                text = ' '.join([result[1] for result in results if result[2] > 0.5])  # confidence > 50%
                ocr_results.append(text)
                logger.debug(f"EasyOCR result: {text[:100]}...")
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Try PaddleOCR
        if self.paddle_ocr:
            try:
                results = self.paddle_ocr.ocr(processed_img, cls=True)
                if results and results[0]:
                    text = ' '.join([item[1][0] for item in results[0] if item[1][1] > 0.5])
                    ocr_results.append(text)
                    logger.debug(f"PaddleOCR result: {text[:100]}...")
            except Exception as e:
                logger.warning(f"PaddleOCR failed: {e}")
        
        # Try Tesseract as fallback
        if TESSERACT_AVAILABLE and not ocr_results:
            try:
                text = pytesseract.image_to_string(processed_img)
                ocr_results.append(text)
                logger.debug(f"Tesseract result: {text[:100]}...")
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")
        
        # Extract names from all OCR results
        for text in ocr_results:
            name = self._extract_name_from_text(text)
            if name:
                return name
        
        return None
    
    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _clean_name(self, name: str) -> str:
        """Clean and normalize extracted name"""
        if not name:
            return ""
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Remove titles
        words = name.split()
        cleaned_words = []
        for word in words:
            if word.upper() not in self.titles_to_remove:
                cleaned_words.append(word)
        
        name = ' '.join(cleaned_words)
        
        # Convert to title case
        name = name.title()
        
        # Remove special characters but keep spaces
        name = re.sub(r'[^\w\s]', '', name)
        
        return name.strip()
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate if extracted text is a valid name"""
        if not name or len(name) < 3:
            return False
        
        # Should have at least 2 words for full name
        words = name.split()
        if len(words) < 2:
            return False
        
        # Each word should be at least 2 characters
        if any(len(word) < 2 for word in words):
            return False
        
        # Should not contain numbers
        if re.search(r'\d', name):
            return False
        
        # Should not be common non-name words
        common_words = {'STATEMENT', 'ACCOUNT', 'BANK', 'BRANCH', 'ADDRESS', 'PHONE', 'EMAIL', 'DATE'}
        if any(word.upper() in common_words for word in words):
            return False
        
        return True

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing special characters"""
    if not filename:
        return "unknown"
    
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename)
    
    # Remove or replace special characters
    filename = re.sub(r'[^\w\s-]', '', filename)
    
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    # Limit length
    if len(filename) > 50:
        filename = filename[:50]
    
    return filename or "unknown"

def generate_dynamic_filename(
    file_id: str,
    original_filename: Optional[str] = None,
    extracted_name: Optional[str] = None,
    language: str = 'en',
    file_format: str = 'xlsx'
) -> str:
    """Generate dynamic filename based on requirements"""
    
    # Current timestamp in required format
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Determine the base name
    if extracted_name:
        # Use extracted name if available
        base_name = sanitize_filename(extracted_name)
        logger.info(f"Using extracted name for filename: {base_name}")
    elif original_filename:
        # Use original filename (remove extension)
        base_name = sanitize_filename(os.path.splitext(original_filename)[0])
        logger.info(f"Using original filename for filename: {base_name}")
    else:
        # Fallback to file_id
        base_name = file_id[:8]  # Use first 8 characters of file_id
        logger.info(f"Using file_id for filename: {base_name}")
    
    # Construct filename: [name]_[timestamp].[extension]
    filename = f"{base_name}_{timestamp}.{file_format}"
    
    logger.info(f"Generated dynamic filename: {filename}")
    return filename

def get_original_filename(file_id: str, upload_dir: str = "uploads") -> Optional[str]:
    """Get original filename from file_id by looking in uploads directory"""
    try:
        if not os.path.exists(upload_dir):
            return None
        
        # Look for file with matching file_id
        for filename in os.listdir(upload_dir):
            if filename.startswith(file_id):
                # Extract original filename pattern if it exists
                # The uploaded file is stored as {file_id}{extension}
                # We need to find the original name from somewhere else
                # For now, we'll extract it from the file_id pattern
                return filename
        
        return None
    except Exception as e:
        logger.error(f"Error getting original filename: {e}")
        return None 