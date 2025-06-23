# PDF Tally Converter

A web application for converting PDF documents containing financial data into structured formats.

## Tech Stack

### Frontend
- React 18 with TypeScript
- Vite for build tooling
- TailwindCSS for styling
- React PDF for PDF rendering
- Axios for API requests
- React Router for navigation
- Heroicons for icons
- React Hot Toast for notifications

### Backend
- Python FastAPI
- PDF processing libraries (pdf2image, pytesseract)
- OpenCV for image processing
- Pandas for data manipulation
- SQLite for data storage

## Features
- PDF and image file upload
- Real-time file preview
- OCR-based text extraction
- Financial data parsing and validation
- Export to multiple formats (Excel, CSV)
- Data validation and editing interface

## Setup

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Tesseract OCR engine

### Frontend Setup
1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file in the root directory:
```
VITE_API_URL=http://localhost:3001
```

3. Start the development server:
```bash
npm run dev
```

### Backend Setup
1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the backend server:
```bash
uvicorn main:app --reload --port 3001
```

## Development

The application runs on:
- Frontend: http://localhost:5173 (or next available port)
- Backend: http://localhost:3001

## Project Structure
```
pdf-tally-converter/
├── src/                    # Frontend source code
│   ├── components/         # React components
│   ├── store/             # State management
│   ├── types/             # TypeScript types
│   └── utils/             # Utility functions
├── backend/               # Backend source code
│   ├── uploads/           # Temporary file storage
│   ├── converted/         # Processed file storage
│   └── main.py           # Main FastAPI application
└── public/               # Static assets
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License 