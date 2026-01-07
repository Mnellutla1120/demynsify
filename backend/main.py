from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import tempfile
import PyPDF2
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import json
from pathlib import Path

# Try to import model predictor (optional)
try:
    from model_predictor import MedicalMisinformationPredictor
    MODEL_PREDICTOR_AVAILABLE = True
except ImportError:
    MODEL_PREDICTOR_AVAILABLE = False
    print("Model predictor not available. Install scikit-learn and joblib to use trained models.")

load_dotenv()

app = FastAPI(title="Deminsify API")

# Get the project root directory (parent of backend)
PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_BUILD_DIR = PROJECT_ROOT / "build"

# CORS middleware - allow all origins when serving from same server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins when serving from same server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client (optional - can use other LLMs)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = None
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)

# Initialize trained model predictor (optional - falls back to OpenAI or mock)
model_predictor = None
use_trained_model = False
if MODEL_PREDICTOR_AVAILABLE:
    model_predictor = MedicalMisinformationPredictor()
    use_trained_model = os.getenv("USE_TRAINED_MODEL", "false").lower() == "true"
    if use_trained_model:
        print("USE_TRAINED_MODEL is enabled. Loading model...")
        success = model_predictor.load_model()
        if success:
            print("✓ Trained model loaded successfully!")
        else:
            print("✗ Failed to load trained model. Falling back to OpenAI/mock.")
    else:
        print("USE_TRAINED_MODEL is not enabled. Using OpenAI API or mock data.")
else:
    print("Model predictor not available. Install scikit-learn and joblib to use trained models.")

class AnalyzeRequest(BaseModel):
    text: str

class URLRequest(BaseModel):
    url: str

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
    return text

def extract_text_from_url(url: str) -> str:
    """Extract text content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:10000]  # Limit to first 10000 chars
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")

def detect_medical_misinformation(text: str) -> dict:
    """
    Analyze text for medical misinformation using trained model, LLM, or fallback.
    Returns a score (0-1, where 1 is high misinformation) and flagged sentences.
    """
    if not text.strip():
        return {
            "misinfo_score": 0.0,
            "accuracy_score": 1.0,
            "flagged_sentences": [],
            "overall_assessment": "No content to analyze"
        }
    
    # Try trained model first if enabled and loaded
    if use_trained_model and model_predictor:
        if not model_predictor.loaded:
            print("Model not loaded, attempting to load now...")
            model_predictor.load_model()
        
        if model_predictor.loaded:
            try:
                print(f"Using trained model to analyze text (length: {len(text)} chars)")
                result = model_predictor.predict_with_details(text)
                if result:
                    print(f"Model prediction: misinfo_score={result.get('misinfo_score', 'N/A')}")
                    return result
                else:
                    print("Model returned None, falling back...")
            except Exception as e:
                print(f"Trained model prediction error: {str(e)}")
                import traceback
                traceback.print_exc()
                # Fall through to OpenAI or mock
        else:
            print("Model failed to load, falling back to OpenAI/mock...")
    
    # Use OpenAI API if available, otherwise return mock data
    if openai_client:
        try:
            prompt = f"""Analyze the following medical/health-related text for misinformation. 
Provide a JSON response with:
1. misinfo_score: float between 0-1 (1 = high misinformation risk)
2. accuracy_score: float between 0-1 (1 = highly accurate)
3. flagged_sentences: array of objects with sentence, label ("False", "Misleading", "Unverified", "Accurate"), and confidence (0-1)
4. overall_assessment: brief summary

Text to analyze:
{text[:4000]}

Respond ONLY with valid JSON in this format:
{{
  "misinfo_score": 0.0-1.0,
  "accuracy_score": 0.0-1.0,
  "flagged_sentences": [
    {{"sentence": "...", "label": "...", "confidence": 0.0-1.0}}
  ],
  "overall_assessment": "..."
}}"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using cheaper model
                messages=[
                    {"role": "system", "content": "You are a medical fact-checking expert. Analyze medical claims for accuracy and misinformation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            # Try to extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            return result
            
        except Exception as e:
            # Fallback to mock if API fails
            print(f"OpenAI API error: {str(e)}")
            pass
    
    # Mock/fallback response
    return {
        "misinfo_score": 0.42,
        "accuracy_score": 0.58,
        "flagged_sentences": [
            {
                "sentence": "Vaccines cause infertility.",
                "label": "False",
                "confidence": 0.91
            }
        ],
        "overall_assessment": "Some claims require verification. Please consult medical professionals."
    }

@app.post("/analyze/text")
async def analyze_text(req: AnalyzeRequest):
    """Analyze text for medical misinformation"""
    result = detect_medical_misinformation(req.text)
    return result

@app.post("/analyze/url")
async def analyze_url(req: URLRequest):
    """Extract content from URL and analyze for medical misinformation"""
    text = extract_text_from_url(req.url)
    result = detect_medical_misinformation(text)
    result["source_url"] = req.url
    result["extracted_text_length"] = len(text)
    return result

@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """Upload and analyze file (PDF or text) for medical misinformation"""
    # Check file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = file.filename.split('.')[-1].lower()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Extract text based on file type
        if file_ext == 'pdf':
            text = extract_text_from_pdf(tmp_path)
        elif file_ext in ['txt', 'text']:
            with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        # Analyze the text
        result = detect_medical_misinformation(text)
        result["filename"] = file.filename
        result["file_type"] = file_ext
        result["text_length"] = len(text)
        
        return result
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}

# Serve React frontend (must be last to catch all non-API routes)
if FRONTEND_BUILD_DIR.exists():
    # Serve static files (JS, CSS, images, etc.)
    app.mount("/static", StaticFiles(directory=FRONTEND_BUILD_DIR / "static"), name="static")
    
    # Serve the React app for all non-API routes (catch-all, must be last)
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """
        Serve React app. This catches all routes not handled by API endpoints.
        """
        # Serve index.html for all routes (React Router will handle client-side routing)
        index_path = FRONTEND_BUILD_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        else:
            raise HTTPException(status_code=404, detail="Frontend not built. Run 'npm run build' first.")
else:
    print(f"⚠️  Frontend build not found at {FRONTEND_BUILD_DIR}")
    print("   Run 'npm run build' in the project root to build the frontend.")
    print("   Or use React dev server separately on port 3000.")
