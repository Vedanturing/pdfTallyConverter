import requests
import os

def test_file_endpoint():
    # Test with a known file ID from the uploads directory
    # Get the first PDF file from the uploads directory
    uploads_dir = "uploads"
    test_file_id = None
    test_file_path = None
    
    for filename in os.listdir(uploads_dir):
        if filename.endswith('.pdf'):
            test_file_id = filename.split('.')[0]
            test_file_path = os.path.join(uploads_dir, filename)
            break
    
    if not test_file_id:
        print("No test file found in uploads directory")
        return
    
    print("\nTesting GET /file/{file_id}")
    print(f"Testing with file ID: {test_file_id}")
    
    # Test the GET endpoint
    response = requests.get(f"http://localhost:3001/file/{test_file_id}")
    
    print(f"Status code: {response.status_code}")
    print(f"Content type: {response.headers.get('content-type')}")
    print(f"Content length: {len(response.content)} bytes")

    # Test the upload endpoint
    print("\nTesting POST /upload")
    with open(test_file_path, 'rb') as f:
        files = {'file': ('test.pdf', f, 'application/pdf')}
        response = requests.post('http://localhost:3001/upload', files=files)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_file_endpoint() 