# Sarcasm Detection Model

A Machine Learning project that detects sarcasm in text using a **Soft-Voting Ensemble approach**. While modern Transformer models are highly capable, this project demonstrates how efficient feature engineering, custom tokenization, and traditional machine learning algorithms can achieve **~86.5% accuracy** with minimal computational overhead.

The repository also includes a **Streamlit Web Application** that compares this traditional ensemble model side-by-side against a fine-tuned Hugging Face Transformer model to highlight performance and latency trade-offs.

## Live Demo
[Live Streamlit App](https://your-app-link.streamlit.app) *(Replace this with your actual Streamlit deployment URL)*

---

## Key Features and Focus Areas
* **No-Transformer Constraint for Ensemble:** Focuses on maximizing the performance of traditional machine learning algorithms.
* **Custom Preprocessing:** Isolates and preserves punctuation marks (such as `!!!` or `...`) as distinct features instead of stripping them, capturing crucial structural cues for sarcasm.
* **Feature Engineering:** Leverages TF-IDF with bigrams (`ngram_range=(1, 2)`) to capture sarcastic word pairs (e.g., *"Yeah right"*).
* **Dual-Model Comparison UI:** The Streamlit app runs the Ensemble model (Logistic Regression, Random Forest, and Gradient Boosting) side-by-side with a deep learning Transformer model (`gnetozela/sarcasm_detection`).

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/beginnerxo/Sarcasm.git
cd Sarcasm
```

### 2. Set Up a Virtual Environment
```bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Streamlit App
To launch the interactive user interface locally:
```bash
streamlit run frontend.py
```

### Training the Model
To retrain the ensemble model from scratch using the datasets in `data/`:
```bash
python train_model.py
```
This script preprocesses the dataset, trains the ensemble classifier, and saves the serialized pipeline to `models/sarcasm_model.pk1`.

### Running Batch Predictions
To generate predictions on a new CSV file:
```bash
python predict_sarcasm.py --input "data/test_data.csv" --output "predictions.csv"
```

---

## Model Architecture

The core ensemble model is a `VotingClassifier` utilizing **soft voting** to combine three distinct estimators:

1. **Logistic Regression (The Baseline):** Captures linear relationships and strong lexical signals (such as words like *"obviously"*).
2. **Random Forest (The Pattern Matcher):** Captures non-linear dependencies and punctuation patterns (100 estimators, max depth of 20).
3. **Gradient Boosting (The Specialist):** Corrects errors from previous trees and handles trickier, edge-case samples.

### Text Processing Pipeline
```text
Input Text ──> Custom Tokenization (Keep Punctuation) ──> TF-IDF Vectorizer (Bigrams) ──> Soft-Voting Ensemble
```

---

## Performance and Metrics

### Ensemble Model
The ensemble model was evaluated on a held-out test set of **1,968 samples**:

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **86.48%** |
| **Test F1-Score** | **86.65%** |

#### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Genuine (0)** | 0.91 | 0.82 | 0.86 | 1,023 |
| **Sarcastic (1)** | 0.82 | 0.91 | 0.87 | 945 |
| **Accuracy** | | | **0.86** | **1,968** |
| **Macro Average** | 0.87 | 0.87 | 0.86 | 1,968 |
| **Weighted Average** | 0.87 | 0.86 | 0.86 | 1,968 |

*Note: The ensemble model performs exceptionally well on structural sarcasm (heavy punctuation usage) but naturally faces challenges with purely contextual sarcasm due to the lack of deep semantic embeddings.*

### Transformer Model
The fine-tuned Transformer model (`distilbert-base-uncased`) was evaluated on the same test set:

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **92.99%** |
| **Test F1-Score** | **92.59%** |


---

## Project Structure

```text
├── data/                  # Training, validation, and test datasets
├── models/                # Saved serialized model pipeline (sarcasm_model.pk1)
├── src/
│   └── utils.py           # Custom preprocessing and tokenization logic
├── frontend.py            # Streamlit web application frontend
├── train_model.py         # Model training script
├── predict_sarcasm.py     # Batch inference script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```
