import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

def validate_numeric(value: str) -> Dict[str, Any]:
    """Validate numeric values and detect common OCR issues"""
    if not value:
        return {"valid": False, "error": "Missing value", "severity": "critical", "fix": None}
    
    # Remove currency symbols and commas
    cleaned = re.sub(r'[₹,]', '', str(value))
    
    # Check for common OCR errors
    ocr_fixes = {
        'O': '0',
        'l': '1',
        'I': '1',
        'S': '5',
        'B': '8',
    }
    
    original = cleaned
    for wrong, correct in ocr_fixes.items():
        cleaned = cleaned.replace(wrong, correct)
    
    try:
        num = float(cleaned)
        if original != cleaned:
            return {
                "valid": True,
                "warning": "Possible OCR error",
                "severity": "warning",
                "fix": cleaned,
                "original": original
            }
        return {"valid": True}
    except ValueError:
        return {
            "valid": False,
            "error": "Invalid numeric value",
            "severity": "critical",
            "fix": cleaned if original != cleaned else None
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
    """Validate entire table data"""
    validation_results = []
    
    for row_idx, row in enumerate(data):
        row_result = {
            "row": row_idx,
            "issues": []
        }
        
        # Validate date
        date_validation = validate_date(str(row.get("DATE", "")))
        if not date_validation.get("valid"):
            row_result["issues"].append({
                "column": "DATE",
                "type": date_validation.get("error"),
                "severity": date_validation.get("severity"),
                "fix": date_validation.get("fix")
            })
        
        # Validate amount
        amount_validation = validate_numeric(str(row.get("AMOUNT", "")))
        if not amount_validation.get("valid"):
            row_result["issues"].append({
                "column": "AMOUNT",
                "type": amount_validation.get("error"),
                "severity": amount_validation.get("severity"),
                "fix": amount_validation.get("fix")
            })
        elif amount_validation.get("warning"):
            row_result["issues"].append({
                "column": "AMOUNT",
                "type": amount_validation.get("warning"),
                "severity": "warning",
                "fix": amount_validation.get("fix"),
                "original": amount_validation.get("original")
            })
        
        # Validate voucher number
        voucher_validation = validate_voucher_no(str(row.get("VOUCHER NO", "")))
        if not voucher_validation.get("valid"):
            row_result["issues"].append({
                "column": "VOUCHER NO",
                "type": voucher_validation.get("error"),
                "severity": voucher_validation.get("severity"),
                "fix": voucher_validation.get("fix")
            })
        
        # Check for negative amounts
        try:
            amount = float(str(row.get("AMOUNT", "0")).replace(",", ""))
            if amount < 0:
                row_result["issues"].append({
                    "column": "AMOUNT",
                    "type": "Negative amount",
                    "severity": "warning",
                    "fix": None
                })
        except ValueError:
            pass
        
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

def validate_date(date_str: str) -> bool:
    """Validate if a string is a valid date."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def validate_amount(amount: Any) -> bool:
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

def validate_financial_data(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate financial entries against predefined rules.
    Returns a dictionary with validation results and errors.
    """
    errors = []
    
    for idx, entry in enumerate(entries):
        row_errors = []
        
        # Required fields
        required_fields = ['date', 'description', 'amount']
        for field in required_fields:
            if not entry.get(field):
                row_errors.append(f"Missing required field: {field}")
        
        # Date validation
        if entry.get('date') and not validate_date(entry['date']):
            row_errors.append("Invalid date format (should be YYYY-MM-DD)")
        
        # Amount validation
        if entry.get('amount') and not validate_amount(entry['amount']):
            row_errors.append("Invalid amount format")
        
        # Balance validation
        if entry.get('balance') and not validate_amount(entry['balance']):
            row_errors.append("Invalid balance format")
        
        # GSTIN validation (if present)
        if entry.get('gstin') and not validate_gstin(entry['gstin']):
            row_errors.append("Invalid GSTIN format")
        
        # Tax rate validation (if present)
        if entry.get('taxRate') and not validate_tax_rate(entry['taxRate']):
            row_errors.append("Invalid tax rate (should be between 0-100)")
        
        if row_errors:
            errors.append({
                "row": idx + 1,
                "errors": row_errors
            })
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }

def validate_table_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate table data for structure and basic content.
    Returns a dictionary with validation results and errors.
    """
    errors = []
    warnings = []
    
    if not data:
        errors.append("No data provided")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings
        }
    
    # Check for consistent columns
    columns = set(data[0].keys())
    for idx, row in enumerate(data[1:], 1):
        row_columns = set(row.keys())
        if row_columns != columns:
            missing = columns - row_columns
            extra = row_columns - columns
            if missing:
                errors.append(f"Row {idx + 1} is missing columns: {', '.join(missing)}")
            if extra:
                errors.append(f"Row {idx + 1} has extra columns: {', '.join(extra)}")
    
    # Validate required columns
    required_columns = {'date', 'description', 'amount'}
    missing_required = required_columns - columns
    if missing_required:
        errors.append(f"Missing required columns: {', '.join(missing_required)}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def get_validation_summary(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary of validation checks and potential issues.
    """
    summary = {
        "total_rows": len(data),
        "empty_fields": [],
        "numeric_issues": [],
        "date_issues": [],
        "duplicate_entries": []
    }
    
    seen_entries = set()
    
    for idx, row in enumerate(data):
        # Check for empty fields
        empty_fields = [k for k, v in row.items() if not str(v).strip()]
        if empty_fields:
            summary["empty_fields"].append({
                "row": idx + 1,
                "fields": empty_fields
            })
        
        # Check numeric fields
        for field in ['amount', 'balance']:
            if field in row and not validate_amount(row[field]):
                summary["numeric_issues"].append({
                    "row": idx + 1,
                    "field": field,
                    "value": row[field]
                })
        
        # Check date fields
        if 'date' in row and not validate_date(row['date']):
            summary["date_issues"].append({
                "row": idx + 1,
                "value": row['date']
            })
        
        # Check for duplicates
        entry_key = f"{row.get('date')}_{row.get('amount')}_{row.get('description')}"
        if entry_key in seen_entries:
            summary["duplicate_entries"].append({
                "row": idx + 1,
                "entry": {
                    "date": row.get('date'),
                    "amount": row.get('amount'),
                    "description": row.get('description')
                }
            })
        seen_entries.add(entry_key)
    
    return summary 