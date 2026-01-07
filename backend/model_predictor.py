"""
Model predictor for medical misinformation detection using trained model.
"""

import joblib
import numpy as np
from pathlib import Path
import json

# Use relative path - models directory should be in the same directory as this script
MODEL_DIR = Path(__file__).parent / "models"
MODEL_NAME = "medical_misinfo_model"

class MedicalMisinformationPredictor:
    """
    Predictor class for medical misinformation detection using trained model.
    """
    
    def __init__(self, model_dir=MODEL_DIR, model_name=MODEL_NAME):
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.loaded = False
        
    def load_model(self):
        """Load trained model and vectorizer."""
        model_path = self.model_dir / f"{self.model_name}.pkl"
        vectorizer_path = self.model_dir / f"{self.model_name}_vectorizer.pkl"
        
        if not model_path.exists() or not vectorizer_path.exists():
            print(f"Model files not found. Expected:")
            print(f"  - {model_path}")
            print(f"  - {vectorizer_path}")
            return False
        
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.loaded = True
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, text):
        """
        Predict misinformation score for given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict with misinfo_score (0-1) and prediction
        """
        if not self.loaded:
            if not self.load_model():
                return None
        
        if not text or len(text.strip()) < 10:
            return {
                "misinfo_score": 0.0,
                "prediction": "Accurate",
                "confidence": 0.0
            }
        
        # Vectorize text
        text_vec = self.vectorizer.transform([text])
        
        # Get probabilities (more nuanced than binary prediction)
        probabilities = self.model.predict_proba(text_vec)[0]
        
        # Get binary prediction for classification
        prediction = self.model.predict(text_vec)[0]
        
        # Use probability of misinformation class (class 1) as the score
        # probabilities[0] = probability of class 0 (accurate)
        # probabilities[1] = probability of class 1 (misinformation)
        if len(probabilities) > 1:
            misinfo_probability = float(probabilities[1])  # Probability of misinformation
        else:
            misinfo_probability = float(prediction)  # Fallback to binary
        
        # Confidence is the maximum probability
        confidence = float(max(probabilities))
        
        return {
            "misinfo_score": misinfo_probability,  # Use probability, not binary
            "accuracy_score": 1.0 - misinfo_probability,
            "prediction": "Misinformation" if prediction == 1 else "Accurate",
            "confidence": confidence
        }
    
    def predict_with_details(self, text):
        """
        Predict with additional details including sentence-level analysis.
        """
        base_prediction = self.predict(text)
        
        if base_prediction is None:
            return None
        
        # Simple sentence-level analysis
        sentences = text.split('.')
        flagged_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Only analyze substantial sentences
                sent_pred = self.predict(sentence)
                if sent_pred and sent_pred['misinfo_score'] > 0.5:
                    flagged_sentences.append({
                        "sentence": sentence.strip(),
                        "label": "False" if sent_pred['misinfo_score'] > 0.7 else "Misleading",
                        "confidence": sent_pred['confidence']
                    })
        
        result = {
            "misinfo_score": base_prediction['misinfo_score'],
            "accuracy_score": base_prediction['accuracy_score'],
            "flagged_sentences": flagged_sentences[:5],  # Limit to top 5
            "overall_assessment": self._generate_assessment(base_prediction)
        }
        
        return result
    
    def _generate_assessment(self, prediction):
        """Generate overall assessment based on prediction."""
        if prediction['misinfo_score'] > 0.7:
            return "High risk of medical misinformation detected. Please verify claims with authoritative medical sources."
        elif prediction['misinfo_score'] > 0.4:
            return "Moderate risk of misinformation. Some claims may require verification."
        else:
            return "Content appears to be generally accurate, but always consult medical professionals for health advice."


