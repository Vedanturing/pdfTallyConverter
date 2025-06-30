import pandas as pd
import xml.etree.ElementTree as ET
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import io

logger = logging.getLogger(__name__)

class TallyParser:
    """Parser for Tally documents in XML and TXT formats"""
    
    def __init__(self):
        # Define the relevant tags to extract from Tally XML
        self.xml_tags = {
            'DATE': ['DATE', 'VOUCHERDATE', 'VOUCHER_DATE'],
            'AMOUNT': ['AMOUNT', 'VOUCHERAMOUNT', 'VOUCHER_AMOUNT', 'LEDGERAMOUNT'],
            'VOUCHERTYPENAME': ['VOUCHERTYPENAME', 'VOUCHER_TYPE', 'VOUCHERTYPE'],
            'PARTYLEDGERNAME': ['PARTYLEDGERNAME', 'PARTY_LEDGER', 'LEDGERNAME', 'PARTY_NAME'],
            'NARRATION': ['NARRATION', 'DESCRIPTION', 'VOUCHERNARRATION', 'REMARKS']
        }
        
        # Common delimiters for TXT files
        self.txt_delimiters = ['\t', '|', ',', ';']
        
    def parse_xml_file(self, file_path: str) -> pd.DataFrame:
        """Parse Tally XML file and extract relevant data"""
        try:
            logger.info(f"Parsing XML file: {file_path}")
            
            # Parse the XML file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract data from XML
            data = []
            vouchers = self._find_vouchers(root)
            
            for voucher in vouchers:
                row_data = self._extract_voucher_data(voucher)
                if row_data:
                    data.append(row_data)
            
            if not data:
                logger.warning("No voucher data found in XML file")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Clean and standardize the data
            df = self._clean_dataframe(df)
            
            logger.info(f"Successfully parsed XML file with {len(df)} records")
            return df
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            raise ValueError(f"Invalid XML file: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing XML file: {str(e)}")
            raise ValueError(f"Error parsing XML file: {str(e)}")
    
    def parse_txt_file(self, file_path: str) -> pd.DataFrame:
        """Parse Tally TXT file with various delimiters"""
        try:
            logger.info(f"Parsing TXT file: {file_path}")
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect the best delimiter
            delimiter = self._detect_delimiter(content)
            logger.info(f"Detected delimiter: '{delimiter}'")
            
            # Parse the content
            df = self._parse_delimited_content(content, delimiter)
            
            # Clean and standardize the data
            df = self._clean_dataframe(df)
            
            logger.info(f"Successfully parsed TXT file with {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing TXT file: {str(e)}")
            raise ValueError(f"Error parsing TXT file: {str(e)}")
    
    def _find_vouchers(self, root: ET.Element) -> List[ET.Element]:
        """Find all voucher elements in the XML tree"""
        vouchers = []
        
        # Common voucher tag names in Tally XML
        voucher_tags = ['VOUCHER', 'TALLYMESSAGE', 'ENVELOPE']
        
        def search_recursive(element):
            # Check if current element is a voucher
            if element.tag.upper() in voucher_tags:
                vouchers.append(element)
            
            # Recursively search children
            for child in element:
                search_recursive(child)
        
        search_recursive(root)
        
        # If no vouchers found, treat each direct child as potential data
        if not vouchers:
            for child in root:
                if len(child) > 0:  # Has sub-elements
                    vouchers.append(child)
        
        return vouchers
    
    def _extract_voucher_data(self, voucher: ET.Element) -> Optional[Dict[str, Any]]:
        """Extract relevant data from a voucher element"""
        data = {}
        
        def extract_recursive(element, depth=0):
            # Limit recursion depth to avoid infinite loops
            if depth > 10:
                return
            
            # Check if this element contains data we want
            tag_upper = element.tag.upper()
            text_content = element.text.strip() if element.text else ''
            
            # Map XML tags to our standard fields
            for field, possible_tags in self.xml_tags.items():
                if tag_upper in [tag.upper() for tag in possible_tags]:
                    if text_content and field not in data:
                        data[field] = text_content
                        break
            
            # Special handling for amount values in attributes
            if 'amount' in tag_upper.lower() and element.attrib:
                for attr_name, attr_value in element.attrib.items():
                    if 'amount' in attr_name.lower() and attr_value:
                        data['AMOUNT'] = attr_value
                        break
            
            # Recursively search children
            for child in element:
                extract_recursive(child, depth + 1)
        
        extract_recursive(voucher)
        
        # Only return data if we have at least some key fields
        if any(field in data for field in ['DATE', 'AMOUNT', 'VOUCHERTYPENAME']):
            return data
        
        return None
    
    def _detect_delimiter(self, content: str) -> str:
        """Detect the most likely delimiter in the text content"""
        lines = content.strip().split('\n')
        if not lines:
            return '\t'
        
        # Count occurrences of each delimiter in the first few lines
        delimiter_counts = {}
        sample_lines = lines[:min(5, len(lines))]
        
        for delimiter in self.txt_delimiters:
            count = sum(line.count(delimiter) for line in sample_lines)
            if count > 0:
                delimiter_counts[delimiter] = count
        
        # Return the delimiter with highest count, default to tab
        if delimiter_counts:
            return max(delimiter_counts, key=delimiter_counts.get)
        else:
            return '\t'
    
    def _parse_delimited_content(self, content: str, delimiter: str) -> pd.DataFrame:
        """Parse delimited text content into DataFrame"""
        # Use StringIO to simulate file reading
        content_io = io.StringIO(content)
        
        try:
            # Try pandas read_csv with detected delimiter
            df = pd.read_csv(content_io, delimiter=delimiter, encoding='utf-8')
            
            # If successful and has data, return it
            if not df.empty and len(df.columns) > 1:
                return df
                
        except Exception as e:
            logger.warning(f"Failed to parse with pandas: {str(e)}")
        
        # Fallback: manual parsing
        lines = content.strip().split('\n')
        if not lines:
            return pd.DataFrame()
        
        # Use first line as headers if it looks like headers
        first_line_parts = lines[0].split(delimiter)
        has_headers = any(part.strip().isalpha() for part in first_line_parts)
        
        if has_headers:
            headers = [part.strip() for part in first_line_parts]
            data_lines = lines[1:]
        else:
            # Generate generic headers
            num_cols = len(first_line_parts)
            headers = [f'Column_{i+1}' for i in range(num_cols)]
            data_lines = lines
        
        # Parse data rows
        rows = []
        for line in data_lines:
            if line.strip():
                parts = line.split(delimiter)
                # Pad or trim to match header count
                while len(parts) < len(headers):
                    parts.append('')
                parts = parts[:len(headers)]
                rows.append(parts)
        
        return pd.DataFrame(rows, columns=headers)
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the DataFrame"""
        if df.empty:
            return df
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Standardize common column names
        column_mapping = {
            'date': 'DATE',
            'voucher_date': 'DATE',
            'voucherdate': 'DATE',
            'amount': 'AMOUNT',
            'voucher_amount': 'AMOUNT',
            'voucheramount': 'AMOUNT',
            'ledger_amount': 'AMOUNT',
            'voucher_type': 'VOUCHERTYPENAME',
            'vouchertype': 'VOUCHERTYPENAME',
            'party_ledger': 'PARTYLEDGERNAME',
            'partyledger': 'PARTYLEDGERNAME',
            'ledger_name': 'PARTYLEDGERNAME',
            'party_name': 'PARTYLEDGERNAME',
            'description': 'NARRATION',
            'remarks': 'NARRATION',
            'voucher_narration': 'NARRATION'
        }
        
        # Apply column mapping
        new_columns = []
        for col in df.columns:
            col_lower = col.lower().strip()
            mapped_col = column_mapping.get(col_lower, col)
            new_columns.append(mapped_col)
        
        df.columns = new_columns
        
        # Clean date fields
        date_columns = [col for col in df.columns if 'DATE' in col.upper()]
        for date_col in date_columns:
            df[date_col] = df[date_col].apply(self._clean_date)
        
        # Clean amount fields
        amount_columns = [col for col in df.columns if 'AMOUNT' in col.upper()]
        for amount_col in amount_columns:
            df[amount_col] = df[amount_col].apply(self._clean_amount)
        
        # Remove leading/trailing whitespace from string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                # Replace 'nan' strings with empty strings
                df[col] = df[col].replace('nan', '')
        
        return df
    
    def _clean_date(self, date_value: Any) -> str:
        """Clean and standardize date values"""
        if pd.isna(date_value) or not date_value:
            return ''
        
        date_str = str(date_value).strip()
        if not date_str or date_str.lower() == 'nan':
            return ''
        
        # Try to parse common date formats
        date_formats = [
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%d.%m.%Y',
            '%Y.%m.%d',
            '%d-%b-%Y',
            '%Y-%b-%d'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no format matches, return original
        return date_str
    
    def _clean_amount(self, amount_value: Any) -> str:
        """Clean and standardize amount values"""
        if pd.isna(amount_value) or not amount_value:
            return ''  # Return empty string instead of '0.00' for missing values
        
        amount_str = str(amount_value).strip()
        if not amount_str or amount_str.lower() == 'nan':
            return ''  # Return empty string instead of '0.00' for nan values
        
        # Remove currency symbols and spaces
        amount_str = re.sub(r'[₹$€£¥,\s]', '', amount_str)
        
        # Handle negative amounts in parentheses
        if amount_str.startswith('(') and amount_str.endswith(')'):
            amount_str = '-' + amount_str[1:-1]
        
        # Try to convert to float
        try:
            amount_float = float(amount_str)
            return f"{amount_float:.2f}"
        except ValueError:
            return amount_str

# Create a singleton instance
tally_parser = TallyParser() 