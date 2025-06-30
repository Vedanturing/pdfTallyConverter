# Dynamic Filename Generation Feature

This implementation improves the download experience by dynamically generating filenames based on:

1. **PDF name** (from original uploaded filename)
2. **Bearer's name** (extracted using OCR from bank statements)
3. **Current date and time**

## Format

The generated filename follows this pattern:
```
[name or original-filename]_[YYYY-MM-DD_HH-MM].[extension]
```

**Examples:**
- `Raj_Sharma_2025-01-15_14-30.xlsx`
- `bank_statement_2025-01-15_14-30.csv`
- `invoice_data_2025-01-15_14-30.json`

## Implementation Details

### Backend Components

#### 1. Name Extractor (`backend/name_extractor.py`)
- **NameExtractor class**: Extracts bearer names from bank statements using multiple OCR engines
- **OCR Support**: EasyOCR, PaddleOCR, Tesseract (with fallback hierarchy)
- **Name Patterns**: Regex patterns for Indian names, titles, and bank statement formats
- **Image Processing**: Focuses on upper 40% of documents where account holder info is typically located

#### 2. File Metadata Manager (`backend/file_metadata.py`)
- **FileMetadataManager class**: Manages file metadata storage and retrieval
- **Metadata Storage**: JSON files containing original filenames and extracted names
- **Cleanup**: Automatic cleanup of old metadata files (7 days default)

#### 3. Modified Endpoints (`backend/main.py`)
- **Upload endpoint**: Stores original filename in metadata
- **Conversion endpoint**: Extracts bearer names and updates metadata
- **Export endpoint**: Uses dynamic filename generation
- **New API endpoint**: `/api/filename/{file_id}` for frontend access

### Frontend Components

#### 1. Dynamic Filename Hook (`src/hooks/useDynamicFilename.ts`)
- **generateFilename()**: Fetches dynamic filename from backend
- **getFileMetadata()**: Retrieves file metadata for display
- **Error handling**: Fallback to timestamp-based filename
- **Loading states**: Progress indication during API calls

## Name Extraction Process

### 1. Text Extraction (Faster)
- Direct text extraction from PDF using PyMuPDF
- Regex pattern matching for common name formats

### 2. OCR Fallback (If text extraction fails)
- Preprocesses image for better OCR results
- Uses multiple OCR engines for best accuracy:
  - **EasyOCR**: Primary choice for accuracy
  - **PaddleOCR**: Secondary choice for table structure
  - **Tesseract**: Fallback option

### 3. Name Validation
- Filters out non-name words (STATEMENT, ACCOUNT, etc.)
- Validates proper name structure (minimum 2 words, no numbers)
- Removes titles (MR, MRS, DR, etc.)

## Usage Examples

### Backend Usage
```python
from name_extractor import NameExtractor, generate_dynamic_filename
from file_metadata import store_file_metadata

# Extract name from PDF
extractor = NameExtractor()
name = extractor.extract_name_from_pdf("bank_statement.pdf")

# Generate dynamic filename
filename = generate_dynamic_filename(
    file_id="abc123",
    original_filename="bank_statement.pdf",
    extracted_name="Raj Sharma",
    language="en",
    file_format="xlsx"
)
# Result: "Raj_Sharma_2025-01-15_14-30.xlsx"
```

### Frontend Usage
```typescript
import { useDynamicFilename } from '../hooks/useDynamicFilename';

const { generateFilename } = useDynamicFilename();

// Generate filename for download
const filename = await generateFilename(fileId, 'xlsx', 'en');
link.setAttribute('download', filename);
```

## Filename Sanitization

- **Unicode normalization**: NFKD normalization for international characters
- **Special character removal**: Only alphanumeric, spaces, hyphens allowed
- **Space handling**: Spaces converted to underscores
- **Length limiting**: Maximum 50 characters for base name
- **Fallback handling**: Uses file_id if no other name available

## Error Handling

### Backend
- Graceful fallback if OCR engines unavailable
- Metadata storage errors don't break upload process
- Fallback to file_id-based naming if extraction fails

### Frontend
- Network error handling with fallback filenames
- Loading states for better UX
- Error messages for debugging

## Configuration

### OCR Engine Priority
1. EasyOCR (best accuracy)
2. PaddleOCR (good for structured documents)
3. Tesseract (traditional OCR)
4. Simple CV fallback (basic pattern matching)

### Name Patterns Supported
- "Account Holder: Raj Sharma"
- "Mr. Raj Sharma"
- "Dear Mr. Raj Sharma"
- "RAJ SHARMA A/C"
- And more variations...

## Benefits

1. **Improved UX**: Users get meaningful filenames instead of random IDs
2. **Better Organization**: Files are easier to identify and organize
3. **Professional Appearance**: Clean, consistent filename format
4. **Multi-language Support**: Localized filename generation
5. **Robust Fallbacks**: Always produces a valid filename

## Future Enhancements

- Support for more document types (invoices, receipts)
- Additional OCR languages
- User preference for filename format
- Bulk filename generation for multiple files
- Integration with document management systems 