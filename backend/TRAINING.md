# Model Training Guide

This guide explains how to train a medical misinformation detection model using the Monant Medical Misinformation Dataset.

## Dataset Setup

### 1. Obtain the Dataset

The Monant Medical Misinformation Dataset is available from:
- **GitHub**: [kinit-sk/medical-misinformation-dataset](https://github.com/kinit-sk/medical-misinformation-dataset)
- **Full Dataset**: Request access via [Zenodo portal](https://zenodo.org) (requires research institution email)

### 2. Download Dataset Files

You need the following CSV files:
- `articles.csv` - Article content and metadata
- `entity_annotations.csv` - Source reliability and article veracity labels
- `relation_annotations.csv` - Claim presence and stance annotations
- `claims.csv` - Fact-checked claims (optional but recommended)

### 3. Organize Files

Place all CSV files in the `backend/data/` directory:

```
backend/
  data/
    articles.csv
    entity_annotations.csv
    relation_annotations.csv
    claims.csv
```

## Training the Model

### 1. Install Dependencies

Make sure you have all required packages:

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Training Script

```bash
python train_model.py
```

The script will:
- Load the dataset from `data/` directory
- Prepare training data (combining articles with labels)
- Split data into 80% training and 20% testing
- Train a logistic regression model (or random forest)
- Evaluate the model
- Save the trained model to `models/` directory

### 3. Training Output

After training, you'll find:
- `models/medical_misinfo_model.pkl` - Trained model
- `models/medical_misinfo_model_vectorizer.pkl` - Text vectorizer
- `models/model_metadata.json` - Training metadata and metrics

## Using the Trained Model

### Option 1: Enable in Backend

Set environment variable to use trained model:

```bash
export USE_TRAINED_MODEL=true
# Or add to .env file:
# USE_TRAINED_MODEL=true
```

Then start the backend server:

```bash
uvicorn main:app --reload --port 8000
```

The backend will use the trained model instead of OpenAI API (if model is loaded successfully).

### Option 2: Use OpenAI API (Fallback)

If `USE_TRAINED_MODEL` is not set or model fails to load, the backend will:
1. Try OpenAI API (if API key is configured)
2. Fall back to mock data

## Model Architecture

### Current Model: Logistic Regression with TF-IDF

- **Vectorization**: TF-IDF with 5000 features
- **N-grams**: Unigrams and bigrams (1-2)
- **Stop words**: English stop words removed
- **Class balancing**: Automatic class weight balancing

### Alternative: Random Forest

To use Random Forest instead, modify `train_model.py`:

```python
model, vectorizer, accuracy = train_model(
    X_train, y_train, X_test, y_test,
    model_type='random_forest'  # Change this
)
```

## Dataset Structure

The Monant dataset includes:

1. **Articles**: News/blog articles about medical topics
2. **Labels**:
   - Source reliability (binary: reliable/unreliable)
   - Article veracity (accurate/misinformation)
   - Claim presence (whether article contains specific claims)
   - Article stance (supporting/opposing/neutral)

3. **Annotations**:
   - Manual labels (expert-verified)
   - Predicted labels (from baseline models)

## Training Parameters

You can modify these in `train_model.py`:

- `TEST_SIZE = 0.2` - 20% for testing (80% training)
- `RANDOM_STATE = 42` - Random seed for reproducibility
- `max_features = 5000` - Number of TF-IDF features
- `ngram_range = (1, 2)` - Unigrams and bigrams

## Evaluation Metrics

The training script outputs:
- **Accuracy**: Overall classification accuracy
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: True/False positives and negatives

## Improving the Model

### 1. More Data
- Use the full dataset from Zenodo (larger than sample)
- Combine with other medical misinformation datasets

### 2. Better Features
- Add domain-specific features (medical terminology)
- Use word embeddings (Word2Vec, GloVe, BERT)
- Include metadata (source credibility, author info)

### 3. Advanced Models
- Fine-tune BERT or other transformer models
- Use ensemble methods
- Implement deep learning models (LSTM, CNN)

### 4. Hyperparameter Tuning
- Use GridSearchCV or RandomizedSearchCV
- Optimize regularization parameters
- Tune feature extraction parameters

## Troubleshooting

### Dataset Not Found
```
Error: Dataset files not found in data/
```
**Solution**: Download dataset files and place in `backend/data/` directory

### Insufficient Data
```
Warning: Only X samples available. Need at least 100 samples.
```
**Solution**: Use the full dataset from Zenodo, not just the sample

### Model Loading Errors
```
Model files not found
```
**Solution**: Run `train_model.py` first to generate model files

### Memory Issues
**Solution**: Reduce `max_features` in TF-IDF vectorizer or use smaller dataset subset

## References

- [Monant Medical Misinformation Dataset](https://github.com/kinit-sk/medical-misinformation-dataset)
- [Paper: Monant Medical Misinformation Dataset: Mapping Articles to Fact-Checked Claims](https://doi.org/10.1145/3477495.3531726)
- [Scikit-learn Documentation](https://scikit-learn.org/)


