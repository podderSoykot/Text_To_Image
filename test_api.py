"""
Test script for the Artistic QR Code API
"""
import requests
import json

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_generate():
    """Test the /generate endpoint"""
    url = f"{BASE_URL}/generate"
    
    # Test request data
    data = {
        "prompt": "A dog in the sky with clouds",
        "qr_data": "https://example.com",
        "image_size": 512,
        "subtlety": 0.92,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "seed": 0,
        "validate_qr": True
    }
    
    print("Testing /generate endpoint...")
    print(f"Request URL: {url}")
    print(f"Request data: {json.dumps(data, indent=2)}")
    print("\nSending request...")
    
    try:
        response = requests.post(url, json=data, timeout=300)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 422:
            print("\n❌ Validation Error (422):")
            print(json.dumps(response.json(), indent=2))
        elif response.status_code == 200:
            print("\n✅ Success!")
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"\n❌ Error ({response.status_code}):")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")

def test_health():
    """Test the /health endpoint"""
    url = f"{BASE_URL}/health"
    print(f"\nTesting /health endpoint...")
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("="*60)
    print("Artistic QR Code API Test")
    print("="*60)
    
    # Test health first
    test_health()
    
    # Test generate
    test_generate()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

