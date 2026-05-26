"""
Test script to verify microservice endpoints are properly configured.
"""

import inspect
from pdf_to_markdown_microservice import (
    health_check, 
    handle_pdf_conversion, 
    handle_hybrid_search,
    handle_store_document
)

def test_endpoints_exist():
    """Test that all required endpoints are defined"""
    
    print("Testing microservice endpoints...")
    
    # Check that functions exist
    endpoints = [
        ('health_check', health_check),
        ('handle_pdf_conversion', handle_pdf_conversion),
        ('handle_hybrid_search', handle_hybrid_search),
        ('handle_store_document', handle_store_document)
    ]
    
    for name, func in endpoints:
        if func is not None:
            print(f"✓ {name} exists")
        else:
            print(f"✗ {name} is missing")
    
    # Check function signatures
    print("\nChecking function signatures...")
    
    # Check health_check
    sig = inspect.signature(health_check)
    print(f"health_check signature: {sig}")
    
    # Check handle_pdf_conversion
    sig = inspect.signature(handle_pdf_conversion)
    print(f"handle_pdf_conversion signature: {sig}")
    
    # Check handle_hybrid_search
    sig = inspect.signature(handle_hybrid_search)
    print(f"handle_hybrid_search signature: {sig}")
    
    # Check handle_store_document
    sig = inspect.signature(handle_store_document)
    print(f"handle_store_document signature: {sig}")
    
    print("\nAll endpoint tests completed!")

if __name__ == "__main__":
    test_endpoints_exist()