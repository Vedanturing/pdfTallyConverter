#!/usr/bin/env python3
"""
Hybrid Bank Statement Parser

A sophisticated parser that combines multiple approaches:
1. Table region detection using deep learning models
2. Cell structure detection
3. Multi-OCR engine support
4. Intelligent row/column reconstruction
5. Schema mapping and data cleaning
6. Fallback to existing parsers
"""

import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import re
import json
from pathlib import Path
import tempfile
import io

# PDF processing
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

# OCR engines
try:
    import paddleocr
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("PaddleOCR not available")

try:
    import easyocr
    EASY_OCR_AVAILABLE = True
except ImportError:
    EASY_OCR_AVAILABLE = False
    logging.warning("EasyOCR not available")

# Deep learning for table detection
try:
    import torch
    import torchvision.transforms as transforms
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Advanced table detection disabled.")

# Fallback imports
try:
    from bank_statement_parser import process_bank_statement, BANK_CONFIGS
    from table_detector import TableDetector
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False
    logging.warning("Fallback parsers not available")

logger = logging.getLogger(__name__)

@dataclass
class TableRegion:
    """Represents a detected table region"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    page_num: int

@dataclass
class CellData:
    """Represents detected cell data"""
    x1: int
    y1: int
    x2: int
    y2: int
    text: str
    confidence: float
    row_idx: int = -1
    col_idx: int = -1

@dataclass
class ParsedTable:
    """Represents a parsed table with structured data"""
    headers: List[str]
    rows: List[Dict[str, str]]
    confidence: float
    region: TableRegion
    missed_areas: List[Dict[str, Any]]

class HybridBankParser:
    """
    Advanced hybrid bank statement parser with multiple detection strategies
    """
    
    def __init__(self, enable_advanced_detection: bool = True):
        self.enable_advanced_detection = enable_advanced_detection
        self.logger = logging.getLogger(__name__)
        
        # Initialize OCR engines
        self.paddle_ocr = None
        self.easy_ocr = None
        self.table_transformer = None
        
        # Schema mapping for different bank statement formats
        self.schema_mapping = {
            'date': [
                'date', 'txn_date', 'transaction_date', 'value_date', 'posting_date',
                'tran_date', 'trans_date', 'dt', 'dated', 'voucher_date'
            ],
            'description': [
                'description', 'particulars', 'narration', 'details', 'transaction_details',
                'remarks', 'purpose', 'transaction_remarks', 'cheque_no', 'reference'
            ],
            'debit': [
                'debit', 'withdrawal', 'dr', 'debit_amount', 'withdrawal_amount',
                'amount_dr', 'paid_amount', 'debited', 'outgoing'
            ],
            'credit': [
                'credit', 'deposit', 'cr', 'credit_amount', 'deposit_amount',
                'amount_cr', 'received_amount', 'credited', 'incoming'
            ],
            'balance': [
                'balance', 'running_balance', 'closing_balance', 'available_balance',
                'bal', 'current_balance', 'account_balance'
            ]
        }
        
        # Initialize components
        self._initialize_engines()
        
        # Fallback parsers
        if FALLBACK_AVAILABLE:
            self.fallback_detector = TableDetector()
    
    def _initialize_engines(self):
        """Initialize OCR and detection engines"""
        if PADDLE_AVAILABLE:
            try:
                # Initialize PaddleOCR (preferred for table structure)
                self.paddle_ocr = paddleocr.PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False,
                    use_gpu=False  # Set to False for stability
                )
                self.logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize PaddleOCR: {str(e)}")
        
        if EASY_OCR_AVAILABLE:
            try:
                # Initialize EasyOCR as fallback
                self.easy_ocr = easyocr.Reader(['en'])
                self.logger.info("EasyOCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize EasyOCR: {str(e)}")
        
        # Initialize Table Transformer if available
        if TRANSFORMERS_AVAILABLE and self.enable_advanced_detection:
            try:
                self.table_processor = AutoImageProcessor.from_pretrained(
                    "microsoft/table-transformer-detection"
                )
                self.table_transformer = TableTransformerForObjectDetection.from_pretrained(
                    "microsoft/table-transformer-detection"
                )
                self.logger.info("Table Transformer initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Table Transformer: {str(e)}")
    
    def detect_table_regions(self, image: np.ndarray, page_num: int = 0) -> List[TableRegion]:
        """
        Detect table regions using multiple approaches
        """
        regions = []
        
        # Method 1: Table Transformer (if available)
        if self.table_transformer is not None:
            try:
                transformer_regions = self._detect_with_transformer(image, page_num)
                regions.extend(transformer_regions)
                self.logger.info(f"Table Transformer found {len(transformer_regions)} regions")
            except Exception as e:
                self.logger.warning(f"Table Transformer detection failed: {str(e)}")
        
        # Method 2: Contour-based detection (fallback)
        if not regions:
            try:
                contour_regions = self._detect_with_contours(image, page_num)
                regions.extend(contour_regions)
                self.logger.info(f"Contour detection found {len(contour_regions)} regions")
            except Exception as e:
                self.logger.warning(f"Contour detection failed: {str(e)}")
        
        # Method 3: Line-based detection (final fallback)
        if not regions:
            try:
                line_regions = self._detect_with_lines(image, page_num)
                regions.extend(line_regions)
                self.logger.info(f"Line detection found {len(line_regions)} regions")
            except Exception as e:
                self.logger.warning(f"Line detection failed: {str(e)}")
        
        # Sort by confidence and size
        regions.sort(key=lambda r: (r.confidence, (r.x2-r.x1)*(r.y2-r.y1)), reverse=True)
        
        return regions
    
    def _detect_with_transformer(self, image: np.ndarray, page_num: int) -> List[TableRegion]:
        """Detect tables using Microsoft Table Transformer"""
        regions = []
        
        try:
            # Convert OpenCV image to PIL
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Process with Table Transformer
            inputs = self.table_processor(pil_image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.table_transformer(**inputs)
            
            # Process predictions
            target_sizes = torch.tensor([pil_image.size[::-1]])  # (height, width)
            results = self.table_processor.post_process_object_detection(
                outputs, threshold=0.7, target_sizes=target_sizes
            )[0]
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if label.item() == 0:  # Table class
                    x1, y1, x2, y2 = box.tolist()
                    regions.append(TableRegion(
                        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                        confidence=score.item(), page_num=page_num
                    ))
        
        except Exception as e:
            self.logger.error(f"Table Transformer detection error: {str(e)}")
        
        return regions
    
    def _detect_with_contours(self, image: np.ndarray, page_num: int) -> List[TableRegion]:
        """Detect tables using contour analysis"""
        regions = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            min_area = (image.shape[0] * image.shape[1]) * 0.05  # At least 5% of image
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / float(h)
                
                if area > min_area and 0.5 < aspect_ratio < 10:
                    confidence = min(0.8, area / (image.shape[0] * image.shape[1]))
                    regions.append(TableRegion(
                        x1=x, y1=y, x2=x+w, y2=y+h,
                        confidence=confidence, page_num=page_num
                    ))
        
        except Exception as e:
            self.logger.error(f"Contour detection error: {str(e)}")
        
        return regions
    
    def _detect_with_lines(self, image: np.ndarray, page_num: int) -> List[TableRegion]:
        """Detect tables using line detection"""
        regions = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find contours in the combined mask
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 200 and h > 100:  # Minimum table size
                    regions.append(TableRegion(
                        x1=x, y1=y, x2=x+w, y2=y+h,
                        confidence=0.6, page_num=page_num
                    ))
        
        except Exception as e:
            self.logger.error(f"Line detection error: {str(e)}")
        
        return regions
    
    def extract_cells_from_region(self, image: np.ndarray, region: TableRegion) -> List[CellData]:
        """Extract individual cells from a table region"""
        cells = []
        
        # Crop the table region
        table_img = image[region.y1:region.y2, region.x1:region.x2]
        
        try:
            # Method 1: PaddleOCR with table structure
            if self.paddle_ocr is not None:
                cells = self._extract_cells_paddle(table_img, region)
                if cells:
                    self.logger.info(f"PaddleOCR extracted {len(cells)} cells")
                    return cells
        except Exception as e:
            self.logger.warning(f"PaddleOCR cell extraction failed: {str(e)}")
        
        try:
            # Method 2: EasyOCR fallback
            if self.easy_ocr is not None:
                cells = self._extract_cells_easy(table_img, region)
                if cells:
                    self.logger.info(f"EasyOCR extracted {len(cells)} cells")
                    return cells
        except Exception as e:
            self.logger.warning(f"EasyOCR cell extraction failed: {str(e)}")
        
        # Method 3: Basic OCR with cv2 and simple text detection
        try:
            cells = self._extract_cells_basic(table_img, region)
            self.logger.info(f"Basic extraction found {len(cells)} cells")
        except Exception as e:
            self.logger.warning(f"Basic extraction failed: {str(e)}")
        
        return cells
    
    def _extract_cells_paddle(self, table_img: np.ndarray, region: TableRegion) -> List[CellData]:
        """Extract cells using PaddleOCR"""
        cells = []
        
        try:
            # Use PaddleOCR to detect text regions
            result = self.paddle_ocr.ocr(table_img, cls=True)
            
            if result and result[0]:
                for line in result[0]:
                    if line:
                        bbox, (text, confidence) = line
                        
                        # Convert bbox to absolute coordinates
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        
                        x1, x2 = int(min(x_coords)), int(max(x_coords))
                        y1, y2 = int(min(y_coords)), int(max(y_coords))
                        
                        # Adjust for region offset
                        x1 += region.x1
                        x2 += region.x1
                        y1 += region.y1
                        y2 += region.y1
                        
                        cells.append(CellData(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            text=text.strip(), confidence=confidence
                        ))
        
        except Exception as e:
            self.logger.error(f"PaddleOCR extraction error: {str(e)}")
        
        return cells
    
    def _extract_cells_easy(self, table_img: np.ndarray, region: TableRegion) -> List[CellData]:
        """Extract cells using EasyOCR"""
        cells = []
        
        try:
            results = self.easy_ocr.readtext(table_img)
            
            for (bbox, text, confidence) in results:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                
                # Adjust for region offset
                x1 += region.x1
                x2 += region.x1
                y1 += region.y1
                y2 += region.y1
                
                cells.append(CellData(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    text=text.strip(), confidence=confidence
                ))
        
        except Exception as e:
            self.logger.error(f"EasyOCR extraction error: {str(e)}")
        
        return cells
    
    def _extract_cells_basic(self, table_img: np.ndarray, region: TableRegion) -> List[CellData]:
        """Extract cells using basic CV2 text detection"""
        cells = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create text regions from contours
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (avoid tiny noise)
                if w > 10 and h > 5:
                    # Extract text from this region (simplified - no actual OCR)
                    # This is a fallback when no OCR engines are available
                    cells.append(CellData(
                        x1=x + region.x1, y1=y + region.y1,
                        x2=x + w + region.x1, y2=y + h + region.y1,
                        text="[TEXT]", confidence=0.3  # Low confidence placeholder
                    ))
        
        except Exception as e:
            self.logger.error(f"Basic extraction error: {str(e)}")
        
        return cells
    
    def reconstruct_table(self, cells: List[CellData]) -> ParsedTable:
        """Reconstruct table structure from detected cells"""
        if not cells:
            return ParsedTable([], [], 0.0, None, [])
        
        # Group cells by y-coordinate to form rows
        cells_with_rows = self._assign_rows(cells)
        
        # Group cells by x-coordinate to form columns
        cells_with_cols = self._assign_columns(cells_with_rows)
        
        # Extract headers (typically first row)
        headers = self._extract_headers(cells_with_cols)
        
        # Extract data rows
        rows = self._extract_rows(cells_with_cols, headers)
        
        # Calculate overall confidence
        avg_confidence = sum(cell.confidence for cell in cells) / len(cells) if cells else 0.0
        
        return ParsedTable(
            headers=headers,
            rows=rows,
            confidence=avg_confidence,
            region=None,  # Will be set by caller
            missed_areas=[]
        )
    
    def _assign_rows(self, cells: List[CellData]) -> List[CellData]:
        """Group cells into rows based on y-coordinates"""
        if not cells:
            return cells
        
        # Sort cells by y-coordinate
        sorted_cells = sorted(cells, key=lambda c: c.y1)
        
        # Group cells with similar y-coordinates
        tolerance = 10  # pixels
        current_row = 0
        current_y = sorted_cells[0].y1
        
        for cell in sorted_cells:
            if abs(cell.y1 - current_y) > tolerance:
                current_row += 1
                current_y = cell.y1
            cell.row_idx = current_row
        
        return sorted_cells
    
    def _assign_columns(self, cells: List[CellData]) -> List[CellData]:
        """Group cells into columns based on x-coordinates"""
        if not cells:
            return cells
        
        # Get unique x-coordinates for column boundaries
        x_positions = sorted(set(cell.x1 for cell in cells))
        
        # Assign column indices
        for cell in cells:
            # Find the closest column position
            col_idx = 0
            min_distance = float('inf')
            
            for i, x_pos in enumerate(x_positions):
                distance = abs(cell.x1 - x_pos)
                if distance < min_distance:
                    min_distance = distance
                    col_idx = i
            
            cell.col_idx = col_idx
        
        return cells
    
    def _extract_headers(self, cells: List[CellData]) -> List[str]:
        """Extract column headers from the first row"""
        if not cells:
            return []
        
        # Get cells from the first row
        header_cells = [cell for cell in cells if cell.row_idx == 0]
        
        # Sort by column index
        header_cells.sort(key=lambda c: c.col_idx)
        
        # Extract header text
        headers = [cell.text for cell in header_cells]
        
        return headers
    
    def _extract_rows(self, cells: List[CellData], headers: List[str]) -> List[Dict[str, str]]:
        """Extract data rows from cells"""
        rows = []
        
        if not cells or not headers:
            return rows
        
        # Group cells by row
        row_groups = {}
        for cell in cells:
            if cell.row_idx > 0:  # Skip header row
                if cell.row_idx not in row_groups:
                    row_groups[cell.row_idx] = []
                row_groups[cell.row_idx].append(cell)
        
        # Convert each row group to dictionary
        for row_idx in sorted(row_groups.keys()):
            row_cells = sorted(row_groups[row_idx], key=lambda c: c.col_idx)
            
            row_dict = {}
            for i, header in enumerate(headers):
                if i < len(row_cells):
                    row_dict[header] = row_cells[i].text
                else:
                    row_dict[header] = ""
            
            rows.append(row_dict)
        
        return rows
    
    def map_to_schema(self, parsed_table: ParsedTable) -> ParsedTable:
        """Map detected headers to standard schema"""
        if not parsed_table.headers:
            return parsed_table
        
        # Create mapping from detected headers to schema
        header_mapping = {}
        
        for detected_header in parsed_table.headers:
            detected_lower = detected_header.lower().strip()
            
            # Find best match in schema
            best_match = detected_header  # Default to original
            
            for schema_key, aliases in self.schema_mapping.items():
                for alias in aliases:
                    if alias in detected_lower or detected_lower in alias:
                        best_match = schema_key
                        break
                if best_match == schema_key:
                    break
            
            header_mapping[detected_header] = best_match
        
        # Apply mapping to headers and rows
        new_headers = [header_mapping.get(h, h) for h in parsed_table.headers]
        new_rows = []
        
        for row in parsed_table.rows:
            new_row = {}
            for old_key, value in row.items():
                new_key = header_mapping.get(old_key, old_key)
                new_row[new_key] = value
            new_rows.append(new_row)
        
        return ParsedTable(
            headers=new_headers,
            rows=new_rows,
            confidence=parsed_table.confidence,
            region=parsed_table.region,
            missed_areas=parsed_table.missed_areas
        )
    
    def clean_data(self, parsed_table: ParsedTable) -> ParsedTable:
        """Clean and standardize the extracted data"""
        if not parsed_table.rows:
            return parsed_table
        
        cleaned_rows = []
        
        for row in parsed_table.rows:
            cleaned_row = {}
            
            for key, value in row.items():
                if key.lower() in ['date']:
                    cleaned_row[key] = self._clean_date(value)
                elif key.lower() in ['debit', 'credit', 'balance', 'amount']:
                    cleaned_row[key] = self._clean_amount(value)
                else:
                    cleaned_row[key] = self._clean_text(value)
            
            # Only add non-empty rows
            if any(v.strip() for v in cleaned_row.values() if isinstance(v, str)):
                cleaned_rows.append(cleaned_row)
        
        return ParsedTable(
            headers=parsed_table.headers,
            rows=cleaned_rows,
            confidence=parsed_table.confidence,
            region=parsed_table.region,
            missed_areas=parsed_table.missed_areas
        )
    
    def _clean_date(self, date_str: str) -> str:
        """Clean and standardize date strings"""
        if not date_str or not date_str.strip():
            return ""
        
        # Remove extra whitespace
        date_str = date_str.strip()
        
        # Try different date formats
        date_formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
            "%d %b %Y", "%d %B %Y",
            "%d-%b-%Y", "%d-%B-%Y"
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # If no format matches, return original
        return date_str
    
    def _clean_amount(self, amount_str: str) -> float:
        """Clean and convert amount strings to float"""
        if not amount_str or not amount_str.strip():
            return 0.0
        
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[₹$€£,\s]', '', amount_str.strip())
        
        # Handle negative indicators
        is_negative = False
        if '(' in cleaned and ')' in cleaned:
            is_negative = True
            cleaned = cleaned.replace('(', '').replace(')', '')
        elif cleaned.startswith('-'):
            is_negative = True
            cleaned = cleaned[1:]
        elif cleaned.lower().endswith('dr'):
            is_negative = True
            cleaned = cleaned[:-2]
        
        # Extract numeric value
        numeric_part = re.search(r'[\d.]+', cleaned)
        if numeric_part:
            try:
                value = float(numeric_part.group())
                return -value if is_negative else value
            except ValueError:
                pass
        
        return 0.0
    
    def _clean_text(self, text: str) -> str:
        """Clean text fields"""
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[|\\`~]', '', cleaned)
        
        return cleaned.strip()
    
    def parse_bank_statement(self, pdf_path: str, password: Optional[str] = None) -> Dict[str, Any]:
        """
        Main parsing function that combines all approaches
        """
        self.logger.info(f"Starting hybrid parsing of: {pdf_path}")
        
        all_tables = []
        missed_areas = []
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            if password:
                doc.authenticate(password)
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
                img_data = pix.tobytes("png")
                img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Detect table regions
                regions = self.detect_table_regions(img_array, page_num)
                
                if regions:
                    self.logger.info(f"Page {page_num}: Found {len(regions)} table regions")
                    
                    # Process each region
                    for region in regions:
                        try:
                            # Extract cells from region
                            cells = self.extract_cells_from_region(img_array, region)
                            
                            if cells:
                                # Reconstruct table
                                parsed_table = self.reconstruct_table(cells)
                                parsed_table.region = region
                                
                                # Map to schema and clean data
                                parsed_table = self.map_to_schema(parsed_table)
                                parsed_table = self.clean_data(parsed_table)
                                
                                if parsed_table.rows:
                                    all_tables.append(parsed_table)
                                    self.logger.info(f"Successfully parsed table with {len(parsed_table.rows)} rows")
                        
                        except Exception as e:
                            self.logger.error(f"Error processing region on page {page_num}: {str(e)}")
                            missed_areas.append({
                                'page': page_num,
                                'region': region,
                                'error': str(e)
                            })
                
                else:
                    # No regions detected, mark page as missed
                    self.logger.warning(f"No table regions detected on page {page_num}")
                    missed_areas.append({
                        'page': page_num,
                        'region': None,
                        'error': 'No table regions detected'
                    })
            
            doc.close()
            
            # Combine all tables into single DataFrame
            if all_tables:
                combined_df = self._combine_tables(all_tables)
                
                return {
                    'success': True,
                    'data': combined_df.to_dict('records') if not combined_df.empty else [],
                    'headers': combined_df.columns.tolist() if not combined_df.empty else [],
                    'confidence': sum(t.confidence for t in all_tables) / len(all_tables),
                    'tables_found': len(all_tables),
                    'missed_areas': missed_areas
                }
            else:
                # Fallback to existing parser
                self.logger.info("No tables found with advanced detection, falling back to existing parser")
                return self._fallback_parse(pdf_path, password, missed_areas)
        
        except Exception as e:
            self.logger.error(f"Hybrid parsing failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Fallback to existing parser
            return self._fallback_parse(pdf_path, password, missed_areas)
    
    def _combine_tables(self, tables: List[ParsedTable]) -> pd.DataFrame:
        """Combine multiple parsed tables into single DataFrame"""
        if not tables:
            return pd.DataFrame()
        
        # Use the most complete table as base
        base_table = max(tables, key=lambda t: len(t.headers))
        
        all_rows = []
        for table in tables:
            all_rows.extend(table.rows)
        
        # Create DataFrame
        df = pd.DataFrame(all_rows, columns=base_table.headers)
        
        # Fill missing columns
        for col in df.columns:
            if col not in df.columns:
                df[col] = ""
        
        return df
    
    def _fallback_parse(self, pdf_path: str, password: Optional[str], missed_areas: List) -> Dict[str, Any]:
        """Fallback to existing parsing methods"""
        if FALLBACK_AVAILABLE:
            try:
                self.logger.info("Using fallback parser")
                df = process_bank_statement(pdf_path, password)
                
                if df is not None and not df.empty:
                    return {
                        'success': True,
                        'data': df.to_dict('records'),
                        'headers': df.columns.tolist(),
                        'confidence': 0.5,  # Lower confidence for fallback
                        'tables_found': 1,
                        'missed_areas': missed_areas,
                        'fallback_used': True
                    }
            except Exception as e:
                self.logger.error(f"Fallback parser also failed: {str(e)}")
        
        return {
            'success': False,
            'error': 'All parsing methods failed',
            'missed_areas': missed_areas
        }

# Main parsing function to maintain compatibility
def parse_with_hybrid_engine(pdf_path: str, password: Optional[str] = None, enable_advanced: bool = True) -> pd.DataFrame:
    """
    Main function to parse bank statement using hybrid engine
    Returns pandas DataFrame for compatibility with existing code
    """
    parser = HybridBankParser(enable_advanced_detection=enable_advanced)
    result = parser.parse_bank_statement(pdf_path, password)
    
    if result['success'] and result['data']:
        return pd.DataFrame(result['data'])
    else:
        return pd.DataFrame() 