# Deminsify Backend

Medical misinformation detection API built with FastAPI.

## Setup

1. Create a virtual environment (if not already created):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up OpenAI API key:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to `.env`
   - If no API key is provided, the app will use mock data for testing

4. Run the server:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

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




