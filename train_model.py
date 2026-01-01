import pandas as pd
import os
import joblib
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score



# Append the src directory to import utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from utils import custom_preprocessor
except ImportError:
    print("Error: Could not import custom_preprocessor from src/utils.py")
    sys.exit(1)
    
def load_dataset(path, name):
    "Helper to load and validate a dataset"
    if not os.path.exists(path):
        print(f"Warning : {name} file not found at {path}")
        return None, None
    
    df = pd.read_csv(path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        print(f"Error: {name} dataset ({path}) must have 'text' and 'label' columns.")
        return None, None
    
    #drop missing values
    df = df.dropna(subset=['text', 'label'])
    
    #ensure integer labels, coerce invalid to NaN and drop
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    
    print(f"Loaded {name}: {len(df)} rows")
    return df['text'], df['label']

def train():
    '''--- Configuration ---
        Expected file names in the data folder
    '''
    
    TRAIN_PATH = 'data/train.csv'
    VAL_PATH = 'data/valid.csv'
    TEST_PATH  = 'data/test.csv'
    
    MODEL_DIR = 'models'
    MODEL_PATH =  os.path.join(MODEL_DIR, "sarcasm_model.pk1")
    
    
    print("--- Loading Data ---")
    X_train, y_train = load_dataset(TRAIN_PATH, "Training")
    X_val, y_val = load_dataset(VAL_PATH, "Validation")
    X_test, y_test = load_dataset(TEST_PATH, "Test")  #this is just for report!!!!
    
    if X_train is None:
        print("CRITICAL ERROR: Cannot proceed without training data.")
        return
    
    print("\n--- Initializing Ensemble ---")
    
    
    # Logistic Regression (Baseline)
    lr = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=1000)
    
    
    # Random Forest(Non-Linear)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    
    
    # Gradient Boosting(Confusing ones  ones)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    ''' --------JURY----------- 

        Voting Classifier(Soft Voting)
        
    '''

    ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
            voting='soft' 
    )


    #pipeline construction
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, preprocessor=custom_preprocessor)),
        ('classifier', ensemble)
    ])


    print("\n--- Training Model on Training Set ---")
    pipeline.fit(X_train, y_train)
    
    
    if X_val is not None:
        print("\n--- Evaluation: Validation Set ---")
        val_preds = pipeline.predict(X_val)
        print(f"Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")
        print(f"Validation F1 Score: {f1_score(y_val, val_preds):.4f}")
        print("-" * 30)
        
        
        #evaluation: Test Set (For Report)
    if X_test is not None:
        print("\n--- Evaluation: Test Set (Report) ---")
        test_preds = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print("\nDetailed Test Report:")
        print(classification_report(y_test, test_preds))
        
        
          #save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()