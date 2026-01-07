"""
Training script for medical misinformation detection model using Monant Medical Misinformation Dataset.
This script trains a model with an 90-10 train-test split.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
# Use relative path - data directory should be in the same directory as this script
DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.1  # 90-10 split
RANDOM_STATE = 42

def load_monant_dataset(data_dir):
    """
    Load Monant Medical Misinformation Dataset from CSV files.
    Expected files:
    - articles.csv: Article content
    - sources.csv: Source information and reliability
    - relation_annotations.csv: Claim presence and stance
    - claims.csv: Fact-checked claims
    """
    print("Loading Monant Medical Misinformation Dataset...")
    
    articles_path = data_dir / "articles.csv"
    sources_path = data_dir / "sources.csv"
    relation_annotations_path = data_dir / "relation_annotations.csv"
    claims_path = data_dir / "claims.csv"
    
    if not articles_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {data_dir}. "
            "Please download the dataset from https://github.com/kinit-sk/medical-misinformation-dataset "
            "or request access via Zenodo portal."
        )
    
    # Load articles
    articles_df = pd.read_csv(articles_path)
    print(f"Loaded {len(articles_df)} articles")
    
    # Load sources (for source reliability)
    sources_df = pd.read_csv(sources_path) if sources_path.exists() else pd.DataFrame()
    if not sources_df.empty:
        print(f"Loaded {len(sources_df)} sources")
    
    # Load claims (to check if claims are false)
    claims_df = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    if not claims_df.empty:
        print(f"Loaded {len(claims_df)} claims")
    
    # Load relation annotations (claim presence, stance)
    relation_annotations_df = pd.read_csv(relation_annotations_path) if relation_annotations_path.exists() else pd.DataFrame()
    if not relation_annotations_df.empty:
        print(f"Loaded {len(relation_annotations_df)} relation annotations")
    
    return articles_df, sources_df, claims_df, relation_annotations_df

def prepare_training_data(articles_df, sources_df, claims_df, relation_annotations_df):
    """
    Prepare training data by combining articles with their labels.
    Creates binary classification: misinformation (1) vs accurate (0)
    """
    print("Preparing training data...")
    
    # Extract article text and IDs
    data = []
    
    for idx, article in articles_df.iterrows():
        article_id = article.get('id', idx)
        # Use body or raw_body if available, fallback to title
        body = article.get('body', '') or article.get('raw_body', '') or ''
        title = article.get('title', '')
        text = f"{title} {body}".strip()
        
        if pd.isna(text) or len(text.strip()) < 50:  # Skip very short articles
            continue
        
        # Determine label from source reliability and claim presence
        label = 0  # Default to accurate (0)
        source_id = article.get('source_id', None)
        
        # Check source reliability first
        if not sources_df.empty and source_id is not None:
            source_info = sources_df[sources_df['id'] == source_id]
            if not source_info.empty:
                # If source is marked as unreliable, more likely to be misinformation
                # This is a heuristic - you may need to adjust based on actual source data structure
                source_name = str(source_info.iloc[0].get('name', '')).lower()
                # Common unreliable sources patterns (adjust based on your data)
                unreliable_patterns = ['naturalnews', 'infowars', 'healthranger']
                if any(pattern in source_name for pattern in unreliable_patterns):
                    label = 1  # Likely misinformation
        
        # Check relation annotations for claim presence
        if not relation_annotations_df.empty:
            # Convert article_id to string for comparison (CSV might have different types)
            article_id_str = str(article_id)
            
            article_relations = relation_annotations_df[
                (relation_annotations_df['source_entity_type'] == 'articles') &
                (relation_annotations_df['source_entity_id'].astype(str) == article_id_str)
            ]
            
            for _, rel in article_relations.iterrows():
                rel_type = rel.get('annotation_type_id', '')
                rel_value = rel.get('value', '{}')
                annotation_category = rel.get('annotation_category', '')
                
                try:
                    value_dict = json.loads(rel_value) if isinstance(rel_value, str) else rel_value
                    
                    # Check annotation type - type 2 is typically claim presence
                    # If value indicates presence of a false claim, mark as misinformation
                    if isinstance(value_dict, dict):
                        # Check if claim is present (value == "yes" or similar)
                        claim_value = value_dict.get('value', '').lower()
                        if claim_value in ['yes', 'true', '1']:
                            # If this is a label (not prediction), or if prediction score is high
                            if annotation_category == 'label' or value_dict.get('score', 0) > 0.5:
                                label = 1  # Misinformation
                                break
                except Exception as e:
                    # Skip if JSON parsing fails
                    pass
        
        data.append({
            'text': text.strip(),
            'label': label,
            'article_id': article_id
        })
    
    df = pd.DataFrame(data)
    print(f"Prepared {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def train_model(X_train, y_train, X_test, y_test, model_type='logistic'):
    """
    Train a classification model for medical misinformation detection.
    """
    print(f"\nTraining {model_type} model...")
    
    # Vectorize text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    if model_type == 'logistic':
        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("Fitting model...")
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Accurate', 'Misinformation']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, vectorizer, accuracy

def save_model(model, vectorizer, model_dir, model_name='medical_misinfo_model'):
    """
    Save trained model and vectorizer.
    """
    model_path = model_dir / f"{model_name}.pkl"
    vectorizer_path = model_dir / f"{model_name}_vectorizer.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("Medical Misinformation Detection Model Training")
    print("Using Monant Medical Misinformation Dataset")
    print("=" * 60)
    
    # Load dataset
    try:
        articles_df, sources_df, claims_df, relation_annotations_df = load_monant_dataset(DATA_DIR)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo obtain the dataset:")
        print("1. Visit: https://github.com/kinit-sk/medical-misinformation-dataset")
        print("2. Download sample data from the repository, or")
        print("3. Request full dataset access via Zenodo portal")
        print("4. Place CSV files in the 'data' directory")
        return
    
    # Prepare training data
    df = prepare_training_data(articles_df, sources_df, claims_df, relation_annotations_df)
    
    if len(df) < 100:
        print(f"\nWarning: Only {len(df)} samples available. Need at least 100 samples for training.")
        print("Please ensure you have the full dataset.")
        return
    
    # Split data (80-20)
    print(f"\nSplitting data: {int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test")
    X = df['text'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    model, vectorizer, accuracy = train_model(
        X_train, y_train, X_test, y_test,
        model_type='logistic'  # Can change to 'random_forest'
    )
    
    # Save model
    save_model(model, vectorizer, MODEL_DIR)
    
    # Save training metadata
    metadata = {
        'model_type': 'logistic_regression',
        'accuracy': float(accuracy),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'dataset': 'Monant Medical Misinformation Dataset'
    }
    
    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTraining metadata saved to: {metadata_path}")
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()


