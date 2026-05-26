# PDF to Markdown Microservice

A microservice that converts PDF files to Markdown format using the Marker library with streaming support.

## Overview

This microservice provides a RESTful API for converting PDF documents to Markdown format. It's designed to be used by other systems that need to process PDF files and receive the content in Markdown format.

## Features

- **PDF to Markdown Conversion**: Converts PDF files to structured Markdown format
- **Streaming Support**: Returns content in chunks for large documents
- **Memory Efficient**: Properly manages GPU/CPU memory
- **RESTful API**: Easy to integrate with other systems
- **Health Check**: Monitor service status
- **Error Handling**: Graceful degradation when components fail

## Architecture

The microservice follows these key design principles:

1. **Initialization**: Loads Marker models once at startup for efficiency
2. **Processing**: Uses the same conversion logic as marker_run.py but with streaming
3. **Memory Management**: Properly handles GPU memory with cleanup
4. **Streaming**: Returns Markdown content in chunks for large documents
5. **Error Handling**: Graceful degradation when components fail

## Endpoints

### Health Check
- **GET** `/health` - Check if service is running

### PDF Conversion
- **POST** `/convert` - Convert PDF to Markdown
  - Request: Multipart form data with `pdf_file` field
  - Response: Streaming Markdown content

## Setup

### Prerequisites
- Python 3.7+
- PyTorch
- Marker library
- CUDA support (recommended for performance)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install marker library (if not already installed)
pip install marker
```

### Running the Service
```bash
python microservice_v2.py
```

The service will start on `http://localhost:8080`

## Usage Examples

### Using curl:
```bash
curl -X POST http://localhost:8080/convert \
  -F "pdf_file=@/path/to/document.pdf" \
  -H "Content-Type: multipart/form-data" \
  --output output.md
```

### Using Python client:
```python
import aiohttp
import asyncio

async def convert_pdf():
    async with aiohttp.ClientSession() as session:
        with open('document.pdf', 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('pdf_file', f, filename='document.pdf')
            async with session.post('http://localhost:8080/convert', data=data) as response:
                async for chunk in response.content.iter_chunked(1024):
                    if chunk:
                        print(chunk.decode('utf-8'), end='', flush=True)
```

## Implementation Details

The microservice reuses the core functionality from `marker_run.py`:

1. **Model Loading**: Uses `create_model_dict` with appropriate device and data type
2. **Converter**: Uses `PdfConverter` with optimized configuration
3. **Memory Management**: Proper cleanup of GPU memory and garbage collection
4. **Error Handling**: Graceful handling of out-of-memory errors

## Testing

Run the test script to verify the service works:
```bash
python test_microservice.py
```

## Files Created

1. `microservice_v2.py` - Main microservice implementation
2. `test_microservice.py` - Test script to verify service functionality
3. `README.md` - This documentation file
4. `pdf_client.py` - Client example for testing the service

## Notes

- The service uses a global converter instance for efficiency
- Temporary files are cleaned up after processing
- Memory is managed carefully to handle large documents
- The service is designed to be run as a standalone microservice

## Troubleshooting

### Common Issues

1. **Marker Library Not Found**: Ensure `marker` is installed with `pip install marker`
2. **CUDA Issues**: If CUDA is not available, the service will fall back to CPU
3. **Memory Errors**: Large PDFs may cause out-of-memory errors; the service attempts to recover

### Logging

The service logs important information including:
- Startup and shutdown messages
- Processing status
- Errors and warnings
- Memory management events

## Future Improvements

1. Add authentication and rate limiting
2. Implement more robust error recovery
3. Add support for other document formats
4. Add configuration options for different conversion settings
5. Implement request queuing for high-volume scenarios