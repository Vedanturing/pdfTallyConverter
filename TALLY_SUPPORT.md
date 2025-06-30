# Tally Document Support

This application now supports parsing and converting Tally documents in XML and TXT formats.

## Supported File Types

### XML Files (.xml)
- Tally XML export files
- Automatically extracts the following fields:
  - `DATE` - Transaction date
  - `VOUCHERTYPENAME` - Type of voucher (Payment, Receipt, Sales, etc.)
  - `AMOUNT` - Transaction amount
  - `PARTYLEDGERNAME` - Party/ledger name
  - `NARRATION` - Transaction description/narration

### TXT Files (.txt)
- Tab-delimited or pipe-delimited text files
- Automatically detects delimiter (tab, pipe, comma, semicolon)
- Supports header row detection
- Compatible with Tally text exports

## Features

### Upload and Parse
- Drag and drop XML/TXT files to upload
- Automatic file type detection
- Real-time parsing and data extraction
- Preview of parsed data with first 500 characters (for text files)

### Data Processing
- Automatic data cleaning and standardization
- Date format normalization (YYYY-MM-DD)
- Amount formatting with proper decimal places
- Column name standardization
- Removal of empty rows and data validation

### Export Options
- **Excel (.xlsx)** - Full spreadsheet with formatting
- **CSV (.csv)** - Comma-separated values for data analysis
- Direct download through web interface
- Preserves all parsed data and formatting

## Usage

### Web Interface
1. Visit the file upload page
2. Drag and drop or click to select your XML/TXT file
3. View the parsed data preview
4. Click "Export Excel" or "Export CSV" to download
5. Use the data in Excel, accounting software, or other tools

### Supported Tally XML Structure
The parser looks for these XML tags:
```xml
<VOUCHER>
  <DATE>2024-06-20</DATE>
  <VOUCHERTYPENAME>Payment</VOUCHERTYPENAME>
  <AMOUNT>5000.00</AMOUNT>
  <PARTYLEDGERNAME>Cash Account</PARTYLEDGERNAME>
  <NARRATION>Office supplies payment</NARRATION>
</VOUCHER>
```

### Supported TXT Format
Tab-delimited format with headers:
```
DATE	VOUCHERTYPENAME	AMOUNT	PARTYLEDGERNAME	NARRATION
2024-06-20	Payment	5000.00	Cash Account	Office supplies payment
2024-06-21	Receipt	10000.00	Customer A	Payment received from customer
```

## Technical Details

### Backend Implementation
- **Parser**: `backend/tally_parser.py` - Handles XML and TXT parsing
- **XML Processing**: Uses ElementTree for robust XML parsing
- **Text Processing**: Automatic delimiter detection and pandas integration
- **Data Cleaning**: Standardizes dates, amounts, and column names
- **Export**: Integrated with existing Excel/CSV export functionality

### Frontend Integration
- Updated file upload components to accept XML/TXT files
- Added Tally file preview with content display
- Integrated with existing data validation and export workflow
- Enhanced file type indicators and user feedback

### API Endpoints
- `POST /upload` - Upload XML/TXT files (now supported)
- `POST /convert/{file_id}` - Parse uploaded Tally files
- `GET /api/convert/{file_id}/xlsx` - Export to Excel
- `GET /api/convert/{file_id}/csv` - Export to CSV

## Sample Files

The repository includes sample files for testing:
- `backend/sample_tally_test.xml` - Sample Tally XML export
- `backend/sample_tally_test.txt` - Sample tab-delimited text file

## Error Handling

The parser includes comprehensive error handling for:
- Invalid XML structure
- Missing required fields
- Unsupported file formats
- Encoding issues
- Empty or corrupt files

## Performance

- **Caching**: File processing results are cached for improved performance
- **Streaming**: Large files are processed in chunks
- **Memory Management**: Efficient handling of large datasets
- **Background Processing**: File parsing runs asynchronously

## Integration with Existing Features

The Tally support is fully integrated with existing features:
- ✅ User authentication and history tracking
- ✅ Data validation and error checking
- ✅ Audit logging and activity tracking
- ✅ Export functionality (Excel, CSV)
- ✅ File management and cleanup
- ✅ Responsive web interface

## Limitations

- XML files must follow Tally export structure
- TXT files should be properly delimited
- Maximum file size: 10MB
- Supported encodings: UTF-8, ASCII

## Future Enhancements

- Additional Tally XML schema support
- Advanced data validation rules
- Custom field mapping
- Batch file processing
- Direct Tally integration 