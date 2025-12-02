import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


from dataset_loader import CONDADataset 

def train_tfidf_baseline(train_path, val_path, save_dir='saved_models'):
    """
    Trains a TF-IDF + Logistic Regression pipeline and saves it.
    
    Args:
        train_path (str): Path to CONDA train CSV.
        val_path (str): Path to CONDA validation CSV.
        save_dir (str): Directory to save the trained model.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("--- Loading Data for TF-IDF ---")
    # We use model_type='tfidf' so the dataset returns raw text strings
    # We enable use_context=True to give this baseline the best chance of competing
    train_dataset = CONDADataset(train_path, model_type='tfidf', use_context=True)
    
    # Extract lists of text and labels directly from the dataset object
    X_train = train_dataset.texts
    y_train = train_dataset.labels
    
    print(f"Training data loaded: {len(X_train)} samples")

    # --- Build Pipeline ---
    # Per your Midterm Report , we use 20,000 features
    print("--- Building TF-IDF + LR Pipeline ---")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=20000, 
            ngram_range=(1, 2),  # Unigrams and Bigrams catch phrases like "gg ez"
            stop_words='english' # Optional: Remove if "u", "ur" are important, but standard for baselines
        )),
        ('clf', LogisticRegression(
            solver='liblinear',  # Good for binary classification
            C=1.0,               # Default regularization
            max_iter=1000        # Ensure convergence
        ))
    ])
    
    # --- Train ---
    print("--- Training Model ---")
    pipeline.fit(X_train, y_train)
    
    # --- Save ---
    model_path = os.path.join(save_dir, 'tfidf_baseline.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")
    print("TF-IDF Baseline training complete.")

if __name__ == "__main__":
    # Update these paths to match your actual file locations
    TRAIN_FILE = 'conda_train.csv' 
    VAL_FILE = 'conda_valid.csv'
    
    train_tfidf_baseline(TRAIN_FILE, VAL_FILE)