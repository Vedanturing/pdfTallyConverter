import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
from typing import List, Tuple, Dict, Optional
import logging
import pandas as pd
import re

logger = logging.getLogger(__name__)

class TableDetector:
    def __init__(self):
        self.blacklist_keywords = [
            'statement', 'page', 'bank', 'computer generated', 'branch',
            'account number', 'ifsc', 'micr', 'generated on', 'printed on',
            'this is a system', 'please examine', 'copyright', 'disclaimer'
        ]
        
        # Standard column name mappings
        self.column_mappings = {
            'date': ['date', 'txn date', 'value date', 'transaction date', 'posting date'],
            'description': ['description', 'particulars', 'narration', 'details', 'transaction details'],
            'debit': ['debit', 'withdrawal (dr)', 'withdrawal', 'dr', 'amount (dr)'],
            'credit': ['credit', 'deposit (cr)', 'deposit', 'cr', 'amount (cr)'],
            'balance': ['balance', 'running balance', 'closing balance', 'bal']
        }

    def detect_table_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect table regions in the image using contour detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Binary threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h)
            
            if area > (image.shape[0] * image.shape[1]) * 0.1 and 0.2 < aspect_ratio < 5:
                table_regions.append((x, y, x + w, y + h))
        
        return table_regions

    def extract_text_from_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """Extract text from a region using Tesseract OCR."""
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(roi)
        return text

    def normalize_column_name(self, col_name: str) -> str:
        """Map column names to standard names."""
        col_name = col_name.lower().strip()
        for std_name, variants in self.column_mappings.items():
            if col_name in variants:
                return std_name
        return col_name

    def filter_rows(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Filter out invalid rows and store discarded lines."""
        discarded_lines = []
        
        # Convert date column to datetime
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        except:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            except:
                pass

        # Filter rows with invalid dates
        invalid_dates = df[df['date'].isna()]
        if not invalid_dates.empty:
            discarded_lines.extend(invalid_dates.values.tolist())
        df = df[df['date'].notna()]

        # Filter rows with blacklisted keywords
        pattern = '|'.join(self.blacklist_keywords)
        blacklisted = df[df['description'].str.lower().str.contains(pattern, na=False)]
        if not blacklisted.empty:
            discarded_lines.extend(blacklisted.values.tolist())
        df = df[~df['description'].str.lower().str.contains(pattern, na=False)]

        return df, discarded_lines

    def process_image(self, image: np.ndarray) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Process an image to extract table data."""
        try:
            # Detect table regions
            regions = self.detect_table_regions(image)
            if not regions:
                return None, ["No table regions detected"]

            # Sort regions by size (largest first)
            regions.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
            
            # Process the largest region
            text = self.extract_text_from_region(image, regions[0])
            
            # Split into lines and filter empty lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Identify header row
            header_row = None
            for i, line in enumerate(lines):
                cols = line.lower().split()
                if any(keyword in cols for keyword in ['date', 'description', 'debit', 'credit', 'balance']):
                    header_row = i
                    break
            
            if header_row is None:
                return None, ["Could not identify table headers"]
            
            # Convert text to DataFrame
            data = [line.split() for line in lines[header_row+1:]]
            headers = lines[header_row].split()
            
            # Normalize column names
            headers = [self.normalize_column_name(h) for h in headers]
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)
            
            # Filter and clean data
            df, discarded_lines = self.filter_rows(df)
            
            return df, discarded_lines
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None, [f"Error processing image: {str(e)}"] 