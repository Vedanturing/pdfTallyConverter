import requests
import time
import sys

def test_endpoint(url, endpoint):
    try:
        response = requests.get(f"{url}{endpoint}")
        print(f"\nTesting {endpoint}:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing {endpoint}: {str(e)}")
        return False

def main():
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"Testing server at {base_url}")
    
    endpoints = ["/", "/health"]
    success = True
    
    for endpoint in endpoints:
        if not test_endpoint(base_url, endpoint):
            success = False
            
    if success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    # Wait for server to start
    time.sleep(2)
    main() 