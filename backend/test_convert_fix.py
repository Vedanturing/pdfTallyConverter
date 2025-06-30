#!/usr/bin/env python3
"""
Test script to verify convert and export functionality
"""
import requests
import json
import os
import sys
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000"
TEST_FILE_ID = "20573aaf-0229-4bbe-91f9-576ed5eb9a65"  # Use an existing file ID from metadata

def test_file_endpoint():
    """Test the /file/{file_id} endpoint with convert=true"""
    print("Testing /file/{file_id}?convert=true endpoint...")
    
    try:
        response = requests.get(f"{API_BASE}/file/{TEST_FILE_ID}?convert=true")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ File endpoint successful")
            print(f"Data keys: {list(data.keys())}")
            
            if 'rows' in data or ('data' in data and 'rows' in data['data']):
                rows = data.get('rows', data.get('data', {}).get('rows', []))
                print(f"Number of rows: {len(rows)}")
                
                if len(rows) > 0:
                    print("First row sample:")
                    first_row = rows[0]
                    for key, value in first_row.items():
                        print(f"  {key}: {value}")
                        
                    # Check for amount columns specifically
                    amount_keywords = ['amount', 'amt', 'balance', 'total', 'debit', 'credit']
                    amount_columns = [k for k in first_row.keys() 
                                    if any(keyword in k.lower() for keyword in amount_keywords)]
                    
                    if amount_columns:
                        print("Amount columns found:")
                        for col in amount_columns:
                            print(f"  {col}: {first_row[col]}")
                    else:
                        print("⚠️ No amount columns detected")
                
                return True
            else:
                print("❌ No rows found in response")
                return False
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_convert_endpoint():
    """Test the /convert/{file_id} endpoint"""
    print("\nTesting /convert/{file_id} endpoint...")
    
    try:
        response = requests.post(f"{API_BASE}/convert/{TEST_FILE_ID}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Convert endpoint successful")
            print(f"Response keys: {list(data.keys())}")
            
            if 'rows' in data:
                print(f"Number of converted rows: {len(data['rows'])}")
                return True
            else:
                print("❌ No rows in convert response")
                return False
        else:
            print(f"❌ Convert error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Convert exception: {e}")
        return False

def test_export_endpoints():
    """Test the /api/convert/{file_id}/{format} endpoints"""
    print("\nTesting export endpoints...")
    
    formats = ['xlsx', 'csv', 'xml']
    results = {}
    
    for format_type in formats:
        print(f"Testing {format_type} export...")
        try:
            response = requests.get(f"{API_BASE}/api/convert/{TEST_FILE_ID}/{format_type}")
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                content_length = len(response.content)
                print(f"  ✅ {format_type} export successful - {content_length} bytes")
                results[format_type] = True
            else:
                print(f"  ❌ {format_type} export failed: {response.text}")
                results[format_type] = False
                
        except Exception as e:
            print(f"  ❌ {format_type} export exception: {e}")
            results[format_type] = False
    
    return results

def test_converted_files():
    """Check if converted files exist"""
    print("\nChecking converted files...")
    
    converted_dir = Path("converted")
    if not converted_dir.exists():
        print("❌ Converted directory doesn't exist")
        return False
    
    json_file = converted_dir / f"{TEST_FILE_ID}.json"
    xlsx_file = converted_dir / f"{TEST_FILE_ID}.xlsx"
    
    files_exist = {
        'json': json_file.exists(),
        'xlsx': xlsx_file.exists()
    }
    
    for file_type, exists in files_exist.items():
        if exists:
            print(f"✅ {file_type} file exists")
        else:
            print(f"❌ {file_type} file missing")
    
    # Check content of JSON file if it exists
    if files_exist['json']:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if 'rows' in data:
                print(f"JSON file contains {len(data['rows'])} rows")
                
                # Check first row for amount values
                if len(data['rows']) > 0:
                    first_row = data['rows'][0]
                    amount_keywords = ['amount', 'amt', 'balance', 'total', 'debit', 'credit']
                    
                    for key, value in first_row.items():
                        if any(keyword in key.lower() for keyword in amount_keywords):
                            print(f"Amount column '{key}': {value} (type: {type(value)})")
        except Exception as e:
            print(f"Error reading JSON file: {e}")
    
    return any(files_exist.values())

def list_available_files():
    """List available files for testing"""
    print("Available files in metadata directory:")
    
    metadata_dir = Path("metadata")
    if metadata_dir.exists():
        json_files = list(metadata_dir.glob("*.json"))
        for file_path in json_files[:5]:  # Show first 5 files
            file_id = file_path.stem
            print(f"  {file_id}")
    else:
        print("No metadata directory found")

def main():
    print("=== PDF Tally Converter - Backend Test ===")
    print(f"API Base URL: {API_BASE}")
    print(f"Test File ID: {TEST_FILE_ID}")
    
    # List available files first
    list_available_files()
    
    # Run tests
    tests = [
        ("File Endpoint", test_file_endpoint()),
        ("Convert Endpoint", test_convert_endpoint()),
        ("Converted Files", test_converted_files()),
    ]
    
    export_results = test_export_endpoints()
    for format_type, success in export_results.items():
        tests.append((f"{format_type.upper()} Export", success))
    
    # Summary
    print("\n=== Test Results ===")
    passed = 0
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main() 