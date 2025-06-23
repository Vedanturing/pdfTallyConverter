# Hybrid Bank Statement Parser Setup Guide

This guide will help you set up the advanced hybrid bank statement parser that can handle various bank statement layouts with high accuracy.

## Features

✅ **Advanced Table Detection**: Uses Microsoft Table-Transformer for accurate table region detection  
✅ **Multi-OCR Support**: PaddleOCR, EasyOCR, Tesseract with intelligent fallbacks  
✅ **Schema Mapping**: Automatically maps detected headers to standard fields  
✅ **Layout Agnostic**: Handles various bank statement formats (SBI, ICICI, HDFC, etc.)  
✅ **Graceful Degradation**: Falls back to simpler methods if advanced features fail  
✅ **No Breaking Changes**: Fully compatible with existing export/validation logic  

## Quick Setup (Recommended)

### 1. Install Dependencies

```bash
# Navigate to backend directory
cd backend

# Install all dependencies (this may take a while)
pip install -r requirements.txt
```

### 2. Test the Setup

```bash
# Run the diagnostic script
python debug_conversion.py
```

### 3. Start the Server

```bash
# Start with hybrid parser enabled (default)
uvicorn main:app --reload --port 8000
```

## Advanced Setup (Optional)

### Enable/Disable Features

You can control the hybrid parser using environment variables:

```bash
# Enable hybrid parser (default: true)
export ENABLE_HYBRID_PARSER=true

# Enable advanced table detection (default: true)
export ENABLE_ADVANCED_TABLE_DETECTION=true

# Enable debug mode for troubleshooting
export PARSER_DEBUG=true
```

### GPU Support (Optional)

For faster processing with GPU support:

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install CUDA-enabled PaddlePaddle
pip install paddlepaddle-gpu
```

## Troubleshooting

### Tesseract Not Found Error

If you get "tesseract is not installed" error:

**Windows:**
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH, or
3. The hybrid parser will automatically fall back to PaddleOCR/EasyOCR

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Memory Issues

If you encounter out-of-memory errors:

1. **Disable GPU processing** (in `hybrid_bank_parser.py`):
   ```python
   use_gpu=False  # Already set by default
   ```

2. **Reduce image scale factor** (in `parser_config.py`):
   ```python
   IMAGE_SCALE_FACTOR = 1.5  # Default is 2.0
   ```

3. **Process fewer pages** (in `parser_config.py`):
   ```python
   MAX_PAGES_TO_PROCESS = 5  # Default is 10
   ```

### Import Errors

If you get import errors for advanced features:

1. **Disable advanced features temporarily**:
   ```bash
   export ENABLE_ADVANCED_TABLE_DETECTION=false
   ```

2. **Install missing dependencies individually**:
   ```bash
   pip install torch transformers  # For Table Transformer
   pip install paddlepaddle paddleocr  # For PaddleOCR
   pip install easyocr  # For EasyOCR
   ```

## Configuration Options

### Parser Priority

The system tries parsers in this order:

1. **Hybrid Parser with Table Transformer** (highest accuracy)
2. **Hybrid Parser with Contour Detection** (good accuracy)
3. **PaddleOCR/EasyOCR** (medium accuracy)
4. **Original pdfplumber parser** (basic accuracy)
5. **Simple CV fallback** (lowest accuracy)

### Customizing OCR Engines

Edit `parser_config.py` to change OCR engine priority:

```python
OCR_ENGINE_PRIORITY = [
    'paddleocr',    # Best for table structure
    'easyocr',      # Good fallback
    'tesseract',    # Traditional OCR
    'simple'        # Basic CV fallback
]
```

## Testing Different Bank Formats

The hybrid parser is designed to handle various bank statement layouts:

- **State Bank of India (SBI)**
- **ICICI Bank**
- **HDFC Bank**
- **Axis Bank**
- **And many others**

Upload different bank statement formats to test the parser's adaptability.

## Performance Optimization

### For High-Volume Processing

1. **Enable GPU if available**:
   ```python
   use_gpu=torch.cuda.is_available()
   ```

2. **Batch processing** (for multiple files):
   ```python
   PARALLEL_PROCESSING = True  # In parser_config.py
   ```

3. **Adjust confidence thresholds** for speed vs. accuracy:
   ```python
   MIN_TABLE_CONFIDENCE = 0.7  # Higher = faster but may miss tables
   MIN_CELL_CONFIDENCE = 0.5   # Higher = faster but may miss cells
   ```

## API Usage

### Using the Hybrid Parser Directly

```python
from hybrid_bank_parser import HybridBankParser

parser = HybridBankParser(enable_advanced_detection=True)
result = parser.parse_bank_statement('path/to/statement.pdf')

if result['success']:
    data = result['data']  # List of transaction dictionaries
    headers = result['headers']  # Column names
    confidence = result['confidence']  # Overall confidence score
    missed_areas = result['missed_areas']  # Areas that couldn't be parsed
```

### Via Main API (Automatic Integration)

The hybrid parser is automatically integrated into the existing API endpoints:

- `GET /file/{file_id}?convert=true` - Now uses hybrid parser by default
- `POST /convert/{file_id}` - Enhanced with hybrid parsing
- All existing export and validation endpoints work unchanged

## Missed Rows and Quality Control

The hybrid parser identifies areas it couldn't parse:

```python
{
    "success": True,
    "data": [...],
    "missed_areas": [
        {
            "page": 1,
            "region": {"x1": 100, "y1": 200, "x2": 500, "y2": 250},
            "error": "Low confidence text detection"
        }
    ]
}
```

These can be reviewed manually or processed with alternative methods.

## Fallback Behavior

The system is designed to never completely fail:

1. **Advanced parsing fails** → Falls back to contour detection
2. **Contour detection fails** → Falls back to line detection  
3. **All table detection fails** → Falls back to original parser
4. **Original parser fails** → Returns structured error with suggestions

## Support and Troubleshooting

### Debug Mode

Enable debug mode for detailed logging:

```bash
export PARSER_DEBUG=true
```

This will:
- Save intermediate images to `debug_output/`
- Provide detailed parsing logs
- Show confidence scores for each component

### Log Analysis

Check the server logs for parsing details:

```bash
# Look for these log patterns:
# "Using hybrid bank parser"
# "Table Transformer found X regions"
# "Successfully parsed table with X rows"
# "Falling back to original parser"
```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "No tables detected" | Poor image quality | Increase `IMAGE_SCALE_FACTOR` |
| "Low confidence results" | Complex layout | Lower confidence thresholds |
| "Memory error" | Large PDF files | Reduce `MAX_PAGES_TO_PROCESS` |
| "Import errors" | Missing dependencies | Install specific packages |
| "Slow processing" | CPU-only processing | Enable GPU support |

## Getting Help

1. **Check the logs** for detailed error messages
2. **Run the debug script**: `python debug_conversion.py`
3. **Test with sample files** to isolate issues
4. **Review configuration** in `parser_config.py`

The hybrid parser is designed to be robust and provide useful results even when some components fail. If you encounter persistent issues, the system will fall back to the original parsing logic to ensure continuity. 