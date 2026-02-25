#!/usr/bin/env python3
"""
Test script to verify all three API endpoints are working correctly.
"""
import requests
import json
from pathlib import Path
import time
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"
ENDPOINTS = [
    ("/predict", "Full model (NLP + CV+Audio)", "file"),
    ("/test/nlp", "NLP-only model", "file"),
    ("/predict/cv-audio", "CV+Audio model", "file"),
]

# Find a test video file
TEST_VIDEOS = [
    Path("examples/sample_data/test_video.mp4"),
    Path("examples/sample_data/test.mp4"),
    Path("data/raw/audio_wav/0/test.wav"),
    Path("data/raw/audio_wav/1/test.wav"),
]

def find_test_file():
    """Find a test video/audio file to use"""
    for test_file in TEST_VIDEOS:
        if test_file.exists():
            print(f"✓ Found test file: {test_file}")
            return test_file
    
    # If no predefined test file, create a minimal one
    print("⚠ No test file found. Using dummy test file.")
    # For now, just return None and we'll handle it
    return None

def test_endpoint(endpoint_path, endpoint_name, test_file):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {endpoint_name}")
    print(f"Endpoint: POST {endpoint_path}")
    print(f"File: {test_file}")
    print(f"{'='*60}")
    
    if not test_file.exists():
        print(f"✗ Test file not found: {test_file}")
        return False
    
    try:
        # Prepare the file upload
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'video/mp4')}
            
            # Send request
            url = f"{API_BASE_URL}{endpoint_path}"
            print(f"Sending request to: {url}")
            
            response = requests.post(url, files=files, timeout=300)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Success!")
                print(f"Response structure:")
                print(json.dumps(result, indent=2, ensure_ascii=False)[:500])  # First 500 chars
                
                # Validate response structure
                if endpoint_path == "/predict":
                    if "overall" in result and "models" in result:
                        print(f"✓ Response has expected structure (overall + models)")
                        return True
                    else:
                        print(f"⚠ Response structure unexpected: {list(result.keys())}")
                        return False
                elif endpoint_path == "/test/nlp":
                    if "prediction" in result and "risk_score" in result:
                        print(f"✓ Response has expected structure (prediction + risk_score)")
                        return True
                    else:
                        print(f"⚠ Response structure unexpected: {list(result.keys())}")
                        return False
                elif endpoint_path == "/predict/cv-audio":
                    if "prediction" in result:
                        print(f"✓ Response has expected structure (prediction)")
                        return True
                    else:
                        print(f"⚠ Response structure unexpected: {list(result.keys())}")
                        return False
            else:
                print(f"✗ Request failed with status {response.status_code}")
                print(f"Response: {response.text[:200]}")
                return False
                
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection error: Cannot connect to {API_BASE_URL}")
        print(f"   Make sure the API is running: python run_all.py --auto-yes")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test routine"""
    print("\n" + "="*60)
    print("API ENDPOINT TEST SUITE")
    print("="*60)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"✓ API is running and responding")
        print(f"  Health check: {response.json()}")
    except Exception as e:
        print(f"✗ API is not responding: {e}")
        print(f"  Start the API with: python run_all.py --auto-yes")
        return False
    
    # Find test file
    test_file = find_test_file()
    if test_file is None:
        print("\n⚠ No test file found. Please provide a video/audio file.")
        print("  Expected locations:")
        for tf in TEST_VIDEOS:
            print(f"    - {tf}")
        return False
    
    # Test each endpoint
    results = {}
    for endpoint_path, endpoint_name, _ in ENDPOINTS:
        results[endpoint_path] = test_endpoint(endpoint_path, endpoint_name, test_file)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for endpoint_path, endpoint_name, _ in ENDPOINTS:
        status = "✓ PASS" if results[endpoint_path] else "✗ FAIL"
        print(f"{status}: {endpoint_name} ({endpoint_path})")
    
    print(f"\nTotal: {passed}/{total} endpoints working")
    
    if passed == total:
        print("\n✓ All endpoints are working correctly!")
        return True
    else:
        print("\n✗ Some endpoints are not working. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
