"""
Test script to verify the microservice works with existing marker functionality.
"""

import os
import sys
import subprocess
import time
import requests
import asyncio
import aiohttp

def test_marker_import():
    """Test that marker library can be imported"""
    try:
        from marker.models import create_model_dict
        from marker.converters.pdf import PdfConverter
        print("✓ Marker library imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Marker library import failed: {e}")
        return False

# REVIEW-03 FIX (Item #10): Proper subprocess cleanup with try/finally and kill fallback.
def _cleanup_process(process: subprocess.Popen) -> None:
    """Terminate a subprocess, falling back to kill if terminate fails."""
    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate gracefully, forcing kill...")
            process.kill()
            process.wait(timeout=5)
    except Exception as e:
        print(f"Warning: Error during process cleanup: {e}")


def test_service_startup():
    """Test that service can start"""
    process = None
    try:
        # Try to start the service in background
        process = subprocess.Popen([
            sys.executable, 'microservice_v2.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Give it a moment to start
        time.sleep(3)

        # Check if it's running
        try:
            response = requests.get('http://localhost:8080/health', timeout=5)
            if response.status_code == 200:
                print("✓ Service started successfully")
                return True
            else:
                print("✗ Service started but health check failed")
                return False
        except requests.exceptions.RequestException:
            print("✗ Service failed to start properly")
            return False

    except Exception as e:
        print(f"✗ Error starting service: {e}")
        return False
    finally:
        # REVIEW-03 FIX (Item #10): Guaranteed cleanup via try/finally.
        if process is not None:
            _cleanup_process(process)

def main():
    print("Testing microservice setup...")
    
    # Test 1: Check marker import
    print("\n1. Testing marker library import:")
    marker_ok = test_marker_import()
    
    # Test 2: Check service startup
    print("\n2. Testing service startup:")
    service_ok = test_service_startup()
    
    if marker_ok and service_ok:
        print("\n✓ All tests passed! Microservice is ready to use.")
        return True
    else:
        print("\n✗ Some tests failed.")
        return False

if __name__ == "__main__":
    main()