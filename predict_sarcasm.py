import pandas as pd
import os
import joblib
import sys
import argparse

#append the src directory to import utils.py (needed when loading the model)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def predict_sarcasm(input_path, output_path, model_path='models/sarcasm_model.pk1'):
    """
    Main inference function.
    
    Args:
        input_path: Path to input CSV file with 'text' column
        output_path: Path to output CSV file with 'text' and 'prediction' columns
        model_path: Path to saved model file
        
    """
    #load the model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    #load input data
    print(f"Loading input data from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)
    
    #validate input format
    if 'text' not in df.columns:
        print("Error: Input CSV must have a 'text' column")
        sys.exit(1)
    
    #handle missing values
    df = df.dropna(subset=['text'])
    
    if len(df) == 0:
        print("Error: No valid text data found in input file")
        sys.exit(1)
    
    print(f"Processing {len(df)} text samples...")
    
    #make predictions
    try:
        predictions = model.predict(df['text'])
        print(f"Predictions generated successfully")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
    
    #ensure predictions are integers (0 or 1)
    predictions = predictions.astype(int)
    
    #create output dataframe
    output_df = pd.DataFrame({
        'text': df['text'],
        'prediction': predictions
    })
    
    #save predictions
    try:
        output_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        print(f"Output contains {len(output_df)} rows with columns: {list(output_df.columns)}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
        sys.exit(1)


def main():
    
    
    parser = argparse.ArgumentParser(
        description='Predict sarcasm in text using trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file with text column'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file with text and prediction columns'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/sarcasm_model.pk1',
        help='Path to saved model file (default: models/sarcasm_model.pk1)'
    )
    
    args = parser.parse_args()
    
    #run inference
    predict_sarcasm(args.input, args.output, args.model)


if __name__ == "__main__":
    main()

