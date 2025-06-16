import pdfplumber
import pymupdf as fitz  # PyMuPDF
import pytesseract
import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from PIL import Image

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

def extract_amount(text: str) -> Optional[float]:
    """Extract amount from text, handling various formats."""
    try:
        # Remove any currency symbols and commas
        text = text.replace('₹', '').replace(',', '').strip()
        
        # Handle Dr/Cr indicators
        if text.upper().endswith('DR'):
            text = '-' + text.upper().replace('DR', '').strip()
        elif text.upper().endswith('CR'):
            text = text.upper().replace('CR', '').strip()
        
        # Try to find a number pattern
        amount_pattern = r'(-?\d+\.?\d*)'
        matches = re.findall(amount_pattern, text)
        
        if matches:
            # Get the first number found
            amount = float(matches[0])
            # If it's in a debit column, make it negative
            if 'debit' in text.lower() or 'dr' in text.lower():
                amount = -abs(amount)
            return amount
        
        return None
    except (ValueError, TypeError):
        return None

def process_bank_statement(file_path: str) -> pd.DataFrame:
    """Process bank statement and return structured data."""
    try:
        # Try pdfplumber first
        with pdfplumber.open(file_path) as pdf:
            tables = []
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    tables.append(pd.DataFrame(table[1:], columns=table[0]))
            
            if tables:
                df = pd.concat(tables, ignore_index=True)
                logger.info(f"Successfully extracted table with pdfplumber: {df.shape}")
                return clean_and_standardize_data(df)
        
        # If pdfplumber fails, try PyMuPDF
        doc = fitz.open(file_path)
        text_content = []
        for page in doc:
            text_content.append(page.get_text())
        doc.close()
        
        # Try to parse the text content
        lines = '\n'.join(text_content).split('\n')
        data = []
        current_entry = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to identify date
            date_match = re.search(r'\d{2}[-/]\d{2}[-/]\d{2,4}', line)
            if date_match:
                if current_entry:
                    data.append(current_entry)
                current_entry = {'date': date_match.group()}
                continue
            
            # Try to identify amount
            amount = extract_amount(line)
            if amount is not None:
                current_entry['amount'] = amount
                # Use the rest of the line as description
                desc = line.replace(str(amount), '').strip()
                if desc:
                    current_entry['description'] = desc
                continue
            
            # If no amount found, treat as description
            if 'description' not in current_entry:
                current_entry['description'] = line
        
        # Add the last entry
        if current_entry:
            data.append(current_entry)
        
        if data:
            df = pd.DataFrame(data)
            logger.info(f"Successfully extracted data with PyMuPDF: {df.shape}")
            return clean_and_standardize_data(df)
        
        # If both methods fail, try OCR
        return extract_data_with_ocr(file_path)
        
    except Exception as e:
        logger.error(f"Error processing bank statement: {str(e)}")
        raise

def clean_and_standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the extracted data."""
    try:
        # Standardize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Map common column names
        column_mapping = {
            'date': 'date',
            'value date': 'date',
            'transaction date': 'date',
            'description': 'description',
            'particulars': 'description',
            'narration': 'description',
            'details': 'description',
            'amount': 'amount',
            'withdrawal amt.': 'debit',
            'deposit amt.': 'credit',
            'debit': 'debit',
            'credit': 'credit',
            'balance': 'balance',
            'closing balance': 'balance'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Combine debit and credit into amount if needed
        if 'amount' not in df.columns and 'debit' in df.columns and 'credit' in df.columns:
            df['amount'] = df['credit'].apply(extract_amount).fillna(0) - df['debit'].apply(extract_amount).fillna(0)
        
        # Ensure required columns exist
        required_columns = ['date', 'description', 'amount']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
                df[col] = None
        
        # Clean date format
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
        
        # Clean amounts
        if 'amount' in df.columns:
            df['amount'] = df['amount'].apply(lambda x: extract_amount(str(x)) if pd.notnull(x) else None)
        
        if 'balance' in df.columns:
            df['balance'] = df['balance'].apply(lambda x: extract_amount(str(x)) if pd.notnull(x) else None)
        
        # Remove rows with no amount or date
        df = df.dropna(subset=['date', 'amount'])
        
        # Sort by date
        df = df.sort_values('date')
        
        return df[['date', 'description', 'amount'] + [col for col in df.columns if col not in ['date', 'description', 'amount']]]
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise

def validate_numeric_fields(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Validate numeric fields in the dataframe"""
    validation_results = {
        'errors': [],
        'warnings': []
    }
    
    numeric_columns = ['debit', 'credit', 'balance']
    for col in numeric_columns:
        if col in df.columns:
            # Check for invalid numbers
            invalid_mask = pd.to_numeric(df[col], errors='coerce').isna()
            invalid_rows = df[invalid_mask]
            
            for idx, row in invalid_rows.iterrows():
                validation_results['errors'].append({
                    'row': idx,
                    'column': col,
                    'value': row[col],
                    'message': f'Invalid numeric value in {col}'
                })
            
            # Check for negative balances
            if col == 'balance':
                negative_mask = df[col] < 0
                negative_rows = df[negative_mask]
                
                for idx, row in negative_rows.iterrows():
                    validation_results['warnings'].append({
                        'row': idx,
                        'column': col,
                        'value': row[col],
                        'message': 'Negative balance detected'
                    })
    
    return validation_results 