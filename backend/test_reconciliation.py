import pandas as pd
import requests
import os
import tempfile
from datetime import datetime

def create_test_excel():
    """Create a test Excel file with Books and Portal sheets"""
    
    # Sample Books data
    books_data = {
        'GSTIN': ['27ABCDE1234F1Z5', '27ABCDE1234F1Z5', '29FGHIJ5678K2L6'],
        'Invoice_No': ['INV001', 'INV002', 'INV003'],
        'Invoice_Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'CGST': [100, 200, 150],
        'SGST': [100, 200, 150],
        'IGST': [0, 0, 300]
    }
    
    # Sample Portal data (INV001 matches, INV002 has different amount, INV004 is new)
    portal_data = {
        'GSTIN': ['27ABCDE1234F1Z5', '27ABCDE1234F1Z5', '30KLMNO9012P3Q7'],
        'Invoice_No': ['INV001', 'INV002', 'INV004'],
        'Invoice_Date': ['2024-01-15', '2024-01-16', '2024-01-18'],
        'CGST': [100, 250, 180],  # INV002 has different CGST (250 vs 200)
        'SGST': [100, 250, 180],  # INV002 has different SGST (250 vs 200)
        'IGST': [0, 0, 360]
    }
    
    # Create temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
            pd.DataFrame(books_data).to_excel(writer, sheet_name='Books', index=False)
            pd.DataFrame(portal_data).to_excel(writer, sheet_name='Portal', index=False)
        
        return tmp_file.name

def test_reconciliation():
    """Test the GST reconciliation endpoint"""
    
    print("🧪 Testing GST Reconciliation Endpoint")
    print("=" * 50)
    
    # Create test Excel file
    test_file = create_test_excel()
    print(f"📄 Created test file: {test_file}")
    
    try:
        # Test the reconciliation endpoint
        url = "http://localhost:8000/reconcile"
        
        with open(test_file, 'rb') as f:
            files = {'file': ('test_reconciliation.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            
            print(f"🚀 Sending request to {url}")
            response = requests.post(url, files=files)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Reconciliation successful!")
            print(f"📈 Summary:")
            print(f"   - Books entries: {result['summary']['total_books_entries']}")
            print(f"   - Portal entries: {result['summary']['total_portal_entries']}")
            print(f"   - Matched entries: {result['summary']['matched_entries']}")
            print(f"   - Books unmatched: {result['summary']['books_unmatched']}")
            print(f"   - Portal unmatched: {result['summary']['portal_unmatched']}")
            print(f"   - Match percentage: {result['summary']['match_percentage']}%")
            print(f"📁 Output file: {result['output_file']}")
            
            # Test download endpoint
            download_url = f"http://localhost:8000{result['download_url']}"
            print(f"📥 Testing download from: {download_url}")
            
            download_response = requests.get(download_url)
            if download_response.status_code == 200:
                print("✅ Download successful!")
                print(f"📦 Downloaded {len(download_response.content)} bytes")
            else:
                print(f"❌ Download failed: {download_response.status_code}")
                
        else:
            print(f"❌ Reconciliation failed: {response.status_code}")
            print(f"💬 Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the backend is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"🧹 Cleaned up test file")

if __name__ == "__main__":
    test_reconciliation() 