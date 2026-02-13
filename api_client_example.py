"""
Example client for the Artistic QR Code API
"""

import requests
import json
from typing import Optional

API_URL = "http://localhost:8000"

def generate_qr_code(
    prompt: str,
    qr_data: str,
    image_size: int = 512,
    subtlety: float = 0.92,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
):
    """Generate QR code via API"""
    
    payload = {
        "prompt": prompt,
        "qr_data": qr_data,
        "image_size": image_size,
        "subtlety": subtlety,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "validate_qr": True
    }
    
    print(f"Requesting QR code generation...")
    print(f"  Prompt: {prompt}")
    print(f"  QR Data: {qr_data}")
    
    response = requests.post(f"{API_URL}/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nSuccess: {result['message']}")
        print(f"Job ID: {result['job_id']}")
        print(f"Scannability: {result.get('scannability_level', 'N/A')}")
        print(f"Scannable: {result.get('scannable', 'N/A')}")
        
        if result.get('image_path'):
            image_url = f"{API_URL}{result['image_path']}"
            print(f"\nDownload image: {image_url}")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def validate_qr_code(filename: str, expected_data: Optional[str] = None):
    """Validate QR code via API"""
    
    params = {}
    if expected_data:
        params['expected_data'] = expected_data
    
    response = requests.get(f"{API_URL}/validate/{filename}", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


def check_health():
    """Check API health"""
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        return response.json()
    return None


if __name__ == "__main__":
    # Check health
    print("Checking API health...")
    health = check_health()
    if health:
        print(f"Status: {health['status']}")
        print(f"Pipeline loaded: {health.get('pipeline_loaded', False)}")
    
    # Example: Generate QR code
    print("\n" + "="*60)
    result = generate_qr_code(
        prompt="A beautiful sunset over mountains, cinematic lighting",
        qr_data="https://example.com",
        subtlety=0.90
    )
    
    if result and result.get('image_path'):
        print(f"\n✅ QR code generated successfully!")
        print(f"   Download: {API_URL}{result['image_path']}")

