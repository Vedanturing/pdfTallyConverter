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

def validate_date(value: str) -> Dict[str, Any]:
    """Validate date formats"""
    if not value:
        return {"valid": False, "error": "Missing date", "severity": "critical", "fix": None}
    
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

def get_validation_summary(validation_results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get summary of validation issues"""
    summary = {
        "critical": 0,
        "warning": 0,
        "info": 0
    }
    
    for result in validation_results:
        for issue in result["issues"]:
            summary[issue["severity"]] += 1
    
    return summary 