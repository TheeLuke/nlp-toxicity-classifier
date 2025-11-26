# baseline_tfidf.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from config import Config

def run_tfidf():
    print("\n--- Baseline 1: TF-IDF + Logistic Regression ---")
    
    # 1. Load Data (Same standardized files)
    print("Loading data...")
    try:
        train_df = pd.read_csv(Config.PROCESSED_TRAIN_PATH)
        test_df = pd.read_csv(Config.PROCESSED_TEST_PATH)
    except FileNotFoundError:
        print("Error: Processed CSVs not found. Run 'preprocess.py' first.")
        return

    # 2. Prepare Inputs (Using raw utterance, not context strings)
    X_train = train_df['utterance'].astype(str)
    y_train = train_df['label']
    X_test = test_df['utterance'].astype(str)
    y_test = test_df['label']

    # 3. Vectorize
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train
    print("Training Logistic Regression...")
    # Class weight balanced to match our Weighted Loss strategy in Deep Learning
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=Config.SEED)
    clf.fit(X_train_vec, y_train)

    # 5. Evaluate
    print("Evaluating...")
    preds = clf.predict(X_test_vec)
    
    f1 = f1_score(y_test, preds, pos_label=1)
    prec = precision_score(y_test, preds, pos_label=1)
    rec = recall_score(y_test, preds, pos_label=1)

    print("\n=== TF-IDF RESULTS ===")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    run_tfidf()