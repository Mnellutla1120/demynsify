# How to Run Training

Your data is loaded and ready! Here's how to train the model:

## Option 1: Using the Script (Recommended)

```bash
cd backend
./run_training.sh
```

## Option 2: Manual Steps

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies (if not already installed):**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run training:**
   ```bash
   python train_model.py
   ```

## What to Expect

The training script will:
1. ✅ Load articles from `data/articles.csv`
2. ✅ Load sources from `data/sources.csv`
3. ✅ Load relation annotations from `data/relation_annotations.csv`
4. ✅ Prepare training data (combining articles with labels)
5. ✅ Split data: 80% training, 20% testing
6. ✅ Train logistic regression model
7. ✅ Evaluate and save model to `models/` directory

## Output Files

After successful training, you'll have:
- `models/medical_misinfo_model.pkl` - Trained model
- `models/medical_misinfo_model_vectorizer.pkl` - Text vectorizer
- `models/model_metadata.json` - Training metrics

## Using the Trained Model

Once training is complete:

1. **Enable the model in backend:**
   Add to `backend/.env`:
   ```
   USE_TRAINED_MODEL=true
   ```

2. **Restart backend server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

The API will now use your trained model instead of OpenAI!

## Troubleshooting

**"Module not found" errors:**
→ Run: `pip install -r requirements.txt`

**"Dataset files not found":**
→ Make sure CSV files are in `backend/data/` directory

**Training takes too long:**
→ The script processes all articles. For faster training, you can modify `train_model.py` to limit the number of articles processed.

