import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/sarcasm_transformer"
NUM_EPOCHS = 3
BATCH_SIZE = 16  # Reduced for stability on standard PCs

def load_and_clean_data(path, name):
    """Load, clean, and validate dataset similar to the original training script."""
    print(f"Loading {name} from {path}...")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None

    df = pd.read_csv(path)
    
    # Clean labels (handle 'label' text in label column)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    
    return df

def compute_metrics(eval_pred):
    """Callback to compute metrics during training."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

def main():
    # 1. Prepare Data
    train_df = load_and_clean_data("data/train.csv", "Training")
    val_df = load_and_clean_data("data/valid.csv", "Validation")
    test_df = load_and_clean_data("data/test.csv", "Test")

    if train_df is None: 
        return

    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    # 2. Tokenization
    print(f"\nDownloading Tokenizer ({MODEL_NAME})...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=128)

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Model Setup
    print(f"\nDownloading Model ({MODEL_NAME})...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # 6. Train
    print("\n--- Starting Training (This may take a while) ---")
    trainer.train()

    # 7. Evaluate
    print("\n--- Final Evaluation on Test Set ---")
    results = trainer.evaluate(tokenized_test)
    print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Test F1 Score: {results['eval_f1']:.4f}")

    # 8. Save
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
