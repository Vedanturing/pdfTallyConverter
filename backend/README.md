# PDF Tally Converter Backend

This is the backend service for the PDF Tally Converter application.

## Deploying to Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click on "New +" and select "Web Service"
3. Connect your GitHub repository
4. Configure the following settings:
   - Name: `pdf-tally-converter-api` (or your preferred name)
   - Environment: `Python`
   - Region: Choose the closest to your users
   - Branch: `main`
   - Root Directory: `pdfTallyConverter/backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app`
   - Instance Type: Choose based on your needs (Start with "Free" for testing)

5. Environment Variables to set:
   ```
   PORT=8000
   PYTHON_VERSION=3.9.0
   ```

6. Click "Create Web Service"

## Important Notes

- The free tier will spin down after 15 minutes of inactivity
- The first request after spin down may take a few seconds
- For production, consider using a paid tier for better performance
- Make sure to set up any additional environment variables your app needs

## API Documentation

Once deployed, you can access the API documentation at:
- Swagger UI: `https://your-render-url/docs`
- ReDoc: `https://your-render-url/redoc`

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

The API will be available at `http://localhost:8000`

## Features

- File upload endpoint with MIME type validation
- Automatic UUID generation for uploaded files
- Metadata extraction for PDFs and images
- File listing and individual file info endpoints
- CORS support for frontend integration

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:
   - On Windows: No additional steps needed
   - On Ubuntu/Debian: `sudo apt-get install libmagic1`
   - On macOS: `brew install libmagic`

## Running the Server

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --port 8000
```

The server will start at http://localhost:8000

## API Endpoints

- `POST /api/upload` - Upload a new file (PDF/image)
- `GET /api/files/{file_id}` - Get metadata for a specific file
- `GET /api/files` - List all uploaded files

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 