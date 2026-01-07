# Deminsify

A medical misinformation detection application similar to GPTZero. Upload files, paste URLs, or enter text to analyze medical content for accuracy and misinformation.

## Features

- **Text Analysis**: Paste medical text directly for analysis
- **URL Analysis**: Extract and analyze content from web pages
- **File Upload**: Upload PDF or text files for analysis
- **Misinformation Scoring**: Get a risk score (0-100%) indicating likelihood of misinformation
- **Accuracy Scoring**: Get an accuracy score (0-100%) indicating content reliability
- **Flagged Claims**: See specific sentences flagged with labels (False, Misleading, Unverified, Accurate)
- **Overall Assessment**: Receive a summary assessment of the content

## Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Activate the virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Set up OpenAI API key for real analysis:
   - Create a `.env` file in the `backend` directory
   - Add: `OPENAI_API_KEY=your_openai_api_key_here`
   - If no API key is provided, the app will use mock data for testing

5. Run the backend server:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Install dependencies (from project root):
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

The app will open at `http://localhost:3000`

## Usage

1. Start both the backend and frontend servers
2. Open the app in your browser at `http://localhost:3000`
3. Choose an input method:
   - **Text**: Paste medical text directly
   - **URL**: Enter a URL to analyze
   - **File Upload**: Upload a PDF or text file
4. Click "Analyze" to get results
5. Review the misinformation score, accuracy score, and flagged claims

## API Endpoints

- `POST /analyze/text` - Analyze text for medical misinformation
- `POST /analyze/url` - Extract content from URL and analyze
- `POST /analyze/file` - Upload and analyze PDF or text file
- `GET /health` - Health check endpoint

## Response Format

```json
{
  "misinfo_score": 0.0-1.0,
  "accuracy_score": 0.0-1.0,
  "flagged_sentences": [
    {
      "sentence": "...",
      "label": "False|Misleading|Unverified|Accurate",
      "confidence": 0.0-1.0
    }
  ],
  "overall_assessment": "..."
}
```

## Technology Stack

- **Frontend**: React, CSS3
- **Backend**: FastAPI, Python
- **AI/ML**: OpenAI GPT-4o-mini (optional)
- **File Processing**: PyPDF2, BeautifulSoup4

## Notes

- The application works without an OpenAI API key (uses mock data)
- For production use, configure an OpenAI API key for accurate analysis
- File uploads are limited to PDF and text files
- URL content extraction is limited to the first 10,000 characters


## Potential Next Steps
- Feed more training data
- Continue training the model to discern misinformation from accuracy
