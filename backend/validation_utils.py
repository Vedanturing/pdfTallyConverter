import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

def validate_numeric(value: str) -> Dict[str, Any]:
    """Validate numeric values and detect common OCR issues"""
    if not value:
        return {"valid": False, "error": "Missing value", "severity": "critical", "fix": None}
    
    # Use improved amount parsing
    try:
        parsed_value = parse_amount_value(value)
        
        # Check if parsing changed the original value significantly
        original_str = str(value).strip()
        if original_str and original_str != str(parsed_value):
            return {
                "valid": True,
                "warning": "Value formatting corrected",
                "severity": "info",
                "fix": parsed_value,
                "original": original_str
            }
        
        return {"valid": True, "value": parsed_value}
        
    except Exception as e:
        return {
            "valid": False,
            "error": f"Invalid numeric value: {str(e)}",
            "severity": "critical",
            "fix": None
        }

def validate_date(value: Any) -> Dict[str, Any]:
    """Validate date formats"""
    if not value:
        return {"valid": False, "error": "Missing date", "severity": "critical", "fix": None}
    
    # Handle pandas Timestamp objects
    if hasattr(value, 'strftime'):
        return {
            "valid": True,
            "fix": value.strftime("%Y-%m-%d")
        }
    
    # Convert to string if not already
    value = str(value)
    
    date_formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%Y.%m.%d"
    ]
    
    for fmt in date_formats:
        try:
            datetime.strptime(value, fmt)
            if fmt != "%Y-%m-%d":
                # Convert to standard format
                date_obj = datetime.strptime(value, fmt)
                return {
                    "valid": True,
                    "warning": "Non-standard date format",
                    "severity": "warning",
                    "fix": date_obj.strftime("%Y-%m-%d"),
                    "original": value
                }
            return {"valid": True}
        except ValueError:
            continue
    
    return {
        "valid": False,
        "error": "Invalid date format",
        "severity": "critical",
        "fix": None
    }

def validate_voucher_no(value: str) -> Dict[str, Any]:
    """Validate voucher numbers"""
    if not value:
        return {"valid": False, "error": "Missing voucher number", "severity": "critical", "fix": None}
    
    # Check for common patterns
    if re.match(r'^[A-Za-z0-9-/]+$', value):
        return {"valid": True}
    
    return {
        "valid": False,
        "error": "Invalid voucher number format",
        "severity": "warning",
        "fix": re.sub(r'[^A-Za-z0-9-/]', '', value)
    }

def validate_table_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Smart validation of table data based on detected columns"""
    if not data:
        return []
    
    validation_results = []
    
    # Get smart column mapping
    headers = list(data[0].keys())
    column_mapping = smart_column_mapping(headers)
    
    # Detect amount columns
    amount_columns = detect_amount_columns(data)
    
    # Find date columns
    date_columns = [col for col in headers if 'date' in col.lower() or 'dt' in col.lower()]
    
    for row_idx, row in enumerate(data):
        row_result = {
            "row": row_idx,
            "issues": []
        }
        
        # Validate date columns (only if they exist)
        for date_col in date_columns:
            if date_col in row:
                date_validation = validate_date(row.get(date_col, ""))
                if not date_validation.get("valid"):
                    row_result["issues"].append({
                        "column": date_col,
                        "type": date_validation.get("error"),
                        "severity": date_validation.get("severity"),
                        "fix": date_validation.get("fix")
                    })
        
        # Validate amount columns (only if they exist)
        for amount_col in amount_columns:
            if amount_col in row:
                amount_validation = validate_numeric(str(row.get(amount_col, "")))
                if not amount_validation.get("valid"):
                    row_result["issues"].append({
                        "column": amount_col,
                        "type": amount_validation.get("error"),
                        "severity": amount_validation.get("severity"),
                        "fix": amount_validation.get("fix")
                    })
                elif amount_validation.get("warning"):
                    row_result["issues"].append({
                        "column": amount_col,
                        "type": amount_validation.get("warning"),
                        "severity": "info",
                        "fix": amount_validation.get("fix"),
                        "original": amount_validation.get("original")
                    })
                
                # Check for negative amounts
                try:
                    amount = parse_amount_value(row.get(amount_col, ""))
                    if amount < 0:
                        row_result["issues"].append({
                            "column": amount_col,
                            "type": "Negative amount",
                            "severity": "warning",
                            "fix": None
                        })
                except:
                    pass
        
        # Only validate voucher numbers if they exist
        voucher_columns = [col for col in headers if any(vch in col.lower() for vch in ['voucher', 'vch', 'ref', 'reference'])]
        for voucher_col in voucher_columns:
            if voucher_col in row:
                voucher_validation = validate_voucher_no(str(row.get(voucher_col, "")))
                if not voucher_validation.get("valid"):
                    row_result["issues"].append({
                        "column": voucher_col,
                        "type": voucher_validation.get("error"),
                        "severity": voucher_validation.get("severity"),
                        "fix": voucher_validation.get("fix")
                    })
        
        if row_result["issues"]:
            validation_results.append(row_result)
    
    return validation_results

def get_validation_summary(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary of validation results.
    Can handle both validation results and raw data.
    """
    summary = {
        "critical": 0,
        "warning": 0,
        "info": 0,
        "total_rows": 0,
        "empty_fields": [],
        "numeric_issues": [],
        "date_issues": [],
        "duplicate_entries": []
    }

    # Check if we're dealing with validation results or raw data
    if validation_results and isinstance(validation_results[0], dict) and "issues" in validation_results[0]:
        # Handle validation results format
        for result in validation_results:
            for issue in result.get("issues", []):
                if isinstance(issue, dict) and "severity" in issue:
                    summary[issue["severity"]] += 1
    else:
        # Handle raw data format
        summary["total_rows"] = len(validation_results)
        seen_entries = set()

        for idx, row in enumerate(validation_results):
            if not isinstance(row, dict):
                continue

            # Check for empty fields
            empty_fields = []
            for k, v in row.items():
                if v is None or (isinstance(v, str) and not v.strip()):
                    empty_fields.append(k)
            
            if empty_fields:
                summary["empty_fields"].append({
                    "row": idx + 1,
                    "fields": empty_fields
                })
                summary["warning"] += len(empty_fields)

            # Check numeric fields
            for field in ['amount', 'balance']:
                if field in row:
                    try:
                        float(str(row[field]).replace(',', ''))
                    except (ValueError, TypeError):
                        summary["numeric_issues"].append({
                            "row": idx + 1,
                            "field": field,
                            "value": row[field]
                        })
                        summary["critical"] += 1

            # Check date fields
            if 'date' in row:
                try:
                    if isinstance(row['date'], str):
                        datetime.strptime(row['date'], "%Y-%m-%d")
                    elif isinstance(row['date'], (datetime, pd.Timestamp)):
                        pass  # These are valid date types
                    else:
                        raise ValueError("Invalid date format")
                except (ValueError, TypeError):
                    summary["date_issues"].append({
                        "row": idx + 1,
                        "value": str(row['date'])
                    })
                    summary["critical"] += 1

            # Check for duplicates
            if all(key in row for key in ['date', 'amount', 'description']):
                entry_key = f"{row['date']}_{row['amount']}_{row['description']}"
                if entry_key in seen_entries:
                    summary["duplicate_entries"].append({
                        "row": idx + 1,
                        "entry": {
                            "date": row['date'],
                            "amount": row['amount'],
                            "description": row['description']
                        }
                    })
                    summary["warning"] += 1
                seen_entries.add(entry_key)

    return summary

class ValidationError(Exception):
    pass

def validate_date_simple(date_str: str) -> bool:
    """Validate if a string is a valid date."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def validate_amount_simple(amount: Any) -> bool:
    """Validate if a value is a valid numeric amount."""
    try:
        float(str(amount).replace(',', ''))
        return True
    except (ValueError, TypeError):
        return False

def validate_gstin(gstin: str) -> bool:
    """Validate if a string is a valid GSTIN."""
    pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$'
    return bool(re.match(pattern, str(gstin)))

def validate_tax_rate(rate: Any) -> bool:
    """Validate if a value is a valid tax rate (0-100)."""
    try:
        rate_float = float(str(rate).replace('%', ''))
        return 0 <= rate_float <= 100
    except (ValueError, TypeError):
        return False

def validate_financial_data(entries: List[Dict[str, Any]], strict: bool = False) -> Dict[str, Any]:
    """
    Validate financial entries against predefined rules.
    Returns a dictionary with validation results and errors.
    
    Args:
        entries: List of financial entries to validate
        strict: If True, enforces all required fields. If False, only validates present fields.
    """
    errors = []
    error_count = 0
    
    for idx, entry in enumerate(entries):
        row_errors = []
        
        # In strict mode, check required fields
        if strict:
            required_fields = ['date', 'description', 'amount']
            for field in required_fields:
                if not entry.get(field):
                    row_errors.append(f"Missing required field: {field}")
        
        # Date validation (only if present)
        if entry.get('date'):
            if not validate_date_simple(entry['date']):
                row_errors.append("Invalid date format")
                error_count += 1
        
        # Amount validation (only if present)
        amount_fields = ['amount', 'debit', 'credit', 'balance']
        for field in amount_fields:
            if entry.get(field):
                if not validate_amount_simple(entry[field]):
                    row_errors.append(f"Invalid {field} format")
                    error_count += 1
        
        # GSTIN validation (if present)
        if entry.get('gstin'):
            if not validate_gstin(entry['gstin']):
                row_errors.append("Invalid GSTIN format")
                error_count += 1
        
        # Tax rate validation (if present)
        if entry.get('taxRate'):
            if not validate_tax_rate(entry['taxRate']):
                row_errors.append("Invalid tax rate (should be between 0-100)")
                error_count += 1
        
        if row_errors:
            errors.append({
                "row": idx + 1,
                "errors": row_errors
            })
    
    return {
        "is_valid": error_count == 0,
        "error_count": error_count,
        "errors": errors
    }

def smart_column_mapping(headers: List[str]) -> Dict[str, str]:
    """Smart mapping of column headers to standard names"""
    column_mapping = {}
    
    # Define mapping patterns for different column types
    patterns = {
        'amount': [
            'amount', 'amt', 'transaction_amount', 'trans_amount', 'txn_amount',
            'credit_amount', 'debit_amount', 'withdrawal', 'deposit', 
            'rupees', 'rs', 'inr', 'total', 'sum', 'value', 'money'
        ],
        'date': [
            'date', 'dt', 'transaction_date', 'trans_date', 'txn_date',
            'value_date', 'posting_date', 'entry_date', 'tran_date'
        ],
        'description': [
            'description', 'desc', 'narration', 'particulars', 'details',
            'remarks', 'reference', 'memo', 'note', 'transaction_desc'
        ],
        'voucher_no': [
            'voucher_no', 'voucher', 'vch_no', 'ref_no', 'reference_no',
            'transaction_id', 'txn_id', 'receipt_no', 'invoice_no'
        ],
        'balance': [
            'balance', 'running_balance', 'closing_balance', 'available_balance',
            'current_balance', 'acc_balance'
        ],
        'debit': [
            'debit', 'dr', 'withdrawal', 'debit_amount', 'dr_amount'
        ],
        'credit': [
            'credit', 'cr', 'deposit', 'credit_amount', 'cr_amount'
        ]
    }
    
    # Create reverse mapping for case-insensitive lookup
    header_mapping = {}
    for header in headers:
        header_lower = header.lower().strip().replace(' ', '_').replace('-', '_')
        
        # Find best match
        for standard_name, variations in patterns.items():
            for variation in variations:
                if variation in header_lower or header_lower == variation:
                    header_mapping[header] = standard_name.upper()
                    break
            if header in header_mapping:
                break
        
        # If no specific mapping found, use cleaned header
        if header not in header_mapping:
            header_mapping[header] = header.upper().replace(' ', '_')
    
    return header_mapping

def parse_amount_value(value: Any) -> float:
    """Enhanced amount parsing with better currency symbol handling"""
    if pd.isna(value) or value is None or value == '':
        return 0.0
    
    # Convert to string first
    str_value = str(value).strip()
    
    if not str_value:
        return 0.0
    
    # Remove common currency symbols and formatting
    cleaned = str_value
    
    # Remove currency symbols
    currency_symbols = ['₹', '$', '€', '£', '¥', 'Rs.', 'Rs', 'INR', 'USD']
    for symbol in currency_symbols:
        cleaned = cleaned.replace(symbol, '')
    
    # Handle brackets for negative values (accounting format)
    is_negative = False
    if cleaned.startswith('(') and cleaned.endswith(')'):
        is_negative = True
        cleaned = cleaned[1:-1]
    elif cleaned.startswith('-'):
        is_negative = True
    
    # Remove thousand separators (commas) but preserve decimal points
    cleaned = cleaned.replace(',', '')
    
    # Remove extra spaces and other formatting
    cleaned = cleaned.strip()
    
    # Handle empty or dash values
    if not cleaned or cleaned == '-' or cleaned.lower() == 'nil':
        return 0.0
    
    try:
        # Try to convert to float
        result = float(cleaned)
        return -result if is_negative else result
    except ValueError:
        # If direct conversion fails, try to extract numeric part
        import re
        numeric_match = re.search(r'-?\d+\.?\d*', cleaned)
        if numeric_match:
            result = float(numeric_match.group())
            return -result if is_negative else result
        return 0.0

def detect_amount_columns(data: List[Dict[str, Any]]) -> List[str]:
    """Detect which columns likely contain amount values"""
    if not data:
        return []
    
    amount_columns = []
    headers = list(data[0].keys()) if data else []
    
    for header in headers:
        header_lower = header.lower()
        
        # Check if header name suggests it's an amount column
        amount_keywords = [
            'amount', 'amt', 'balance', 'total', 'sum', 'value', 'money',
            'debit', 'credit', 'withdrawal', 'deposit', 'rupees', 'rs', 'inr'
        ]
        
        if any(keyword in header_lower for keyword in amount_keywords):
            amount_columns.append(header)
            continue
        
        # Check if values in this column are predominantly numeric
        numeric_count = 0
        total_count = 0
        
        for row in data[:10]:  # Check first 10 rows
            value = row.get(header)
            if value is not None and str(value).strip():
                total_count += 1
                try:
                    parse_amount_value(value)
                    numeric_count += 1
                except:
                    pass
        
        # If more than 80% of values are numeric, consider it an amount column
        if total_count > 0 and (numeric_count / total_count) > 0.8:
            amount_columns.append(header)
    
    return amount_columns 