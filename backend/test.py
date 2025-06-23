import requests

def test_api():
    base_url = "http://localhost:8000"
    
    # Test root endpoint
    response = requests.get(f"{base_url}/")
    print("Testing /")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    
    # Test health endpoint
    response = requests.get(f"{base_url}/health")
    print("Testing /health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_api() 