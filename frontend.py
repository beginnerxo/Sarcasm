import streamlit as st
import os
import sys
import time
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer

@st.cache_resource
def load_transformer_model():
    model_path = os.path.join("models", "sarcasm_transformer")
    if os.path.exists(model_path):
        try:
            # Lazy import to speed up initial app load
            import torch
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            
            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
            model.eval() # Set to evaluation mode
            return tokenizer, model
        except Exception as e:
            st.error(f"Error loading transformer: {e}")
            return None, None
    return None, None
import time
import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Add src to path to import utils properly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    import utils
    # CRITICAL FIX: Register 'utils' in sys.modules so pickle finds it
    sys.modules['utils'] = utils
except ImportError:
    st.error("Could not import 'src/utils.py'. Please ensure the file exists.")

# Page config (Must be first st command)
st.set_page_config(
    page_title="Sarcasm Detector Model",
    page_icon="😏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%);
        color: #fafafa;
    }
    
    /* Typography */
    .stTitle {
        background: linear-gradient(90deg, #ff4b4b, #ff8f6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #a0a0a0;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Input Area */
    .stTextArea textarea {
        background-color: rgba(38, 39, 48, 0.7);
        color: #ffffff;
        border-radius: 12px;
        border: 1px solid rgba(255, 75, 75, 0.2);
        backdrop-filter: blur(5px);
        transition: border 0.3s;
    }
    .stTextArea textarea:focus {
        border: 1px solid #ff4b4b;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.2);
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #ff4b4b, #ff3333);
        color: white;
        border-radius: 25px;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* Result Cards */
    .result-box {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2rem;
        animation: slideUp 0.6s ease-out;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #12141a;
        border-right: 1px solid #262730;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sarcasm_model():
    model_path = os.path.join("models", "sarcasm_model.pk1")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_transformer_model():
    try:
        # Load from Hugging Face instead of local folder
        model_name = "gnetozela/sarcasm_detection"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading transformer from Hugging Face: {e}")
        return None, None

def main():
    # Load models on startup
    with st.spinner("Loading AI models..."):
        ensemble_model = load_sarcasm_model()
        trans_tokenizer, trans_model = load_transformer_model()
    
    # ... (Sidebar code remains the same)

    # Main Content
    st.markdown("<h1 class='stTitle'>😏 Sarcasm Detector AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Comparing Ensemble Learning vs. Transformers from Hugging Face</p>", unsafe_allow_html=True)

    # Layout
    col_main, col_dummy = st.columns([2, 0.5]) # Main column takes more space

    with col_main:
        # Helper to handle input state
        if 'input_val' not in st.session_state:
            st.session_state.input_val = ""

        # Quick Examples with streaming effect logic
        st.markdown("Try an example:")
        examples = [
            "Oh great, another meeting.", 
            "I absolutely love being stuck in traffic.", 
            "The weather is lovely today."
        ]
        
        cols = st.columns(len(examples))
        
        # Input Placeholder
        input_area_placeholder = st.empty()

        # When an example is clicked
        for i, ex in enumerate(examples):
            if cols[i].button(ex, key=f"ex_{i}"):
                # Typewriter effect directly in the text area
                typed_text = ""
                for char in ex:
                    typed_text += char
                    # Update the text area with partial text
                    input_area_placeholder.text_area(
                        "Enter your text:",
                        value=typed_text,
                        placeholder="Type something sarcastic here...",
                        height=120,
                        key=f"input_anim_{i}_{len(typed_text)}" # Unique key to force re-render
                    )
                    time.sleep(0.01) # Faster typing for smoother feel
                
                # Final set to session state
                st.session_state.input_val = ex
        
        # Helper to handle input state
        if 'input_val' not in st.session_state:
            st.session_state.input_val = ""

        # Final Render of the Input Area (if not animating)
        user_input = input_area_placeholder.text_area(
            "Enter your text:",
            value=st.session_state.input_val,
            placeholder="Type something sarcastic here...",
            height=120
        )
        
        # Update session state on manual change
        if user_input != st.session_state.input_val:
            st.session_state.input_val = user_input

        # Analyze Button
        if st.button("🔍 Analyze Tone", use_container_width=True):
            if user_input.strip():
                # Models are already loaded on startup
                
                # Simulated "Thinking" Stream
                status_placeholder = st.empty()
                thoughts = ["Tokenizing text...", "Consulting the Random Forest...", "Querying the Transformer...", "Aggregating sarcasm levels..."]
                for thought in thoughts:
                    status_placeholder.markdown(f"*{thought}*")
                    time.sleep(0.2)
                status_placeholder.empty()

                try:
                    # --- 1. Ensemble Prediction ---
                    ens_pred = None
                    ens_conf = None
                    if ensemble_model:
                        ens_pred_label = ensemble_model.predict([user_input])[0]
                        if hasattr(ensemble_model, "predict_proba"):
                            probs = ensemble_model.predict_proba([user_input])[0]
                            ens_conf = max(probs)
                        ens_pred = "Sarcastic" if ens_pred_label == 1 else "Genuine"
                        ens_color = "#ff4b4b" if ens_pred_label == 1 else "#4bff64"

                    # --- 2. Transformer Prediction ---
                    trans_pred = None
                    trans_conf = None
                    if trans_model:
                        # Import torch locally since it's not global
                        import torch
                        inputs = trans_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
                        with torch.no_grad():
                            outputs = trans_model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        trans_conf = torch.max(probs).item()
                        trans_pred_label = torch.argmax(probs, dim=-1).item()
                        
                        trans_pred = "Sarcastic" if trans_pred_label == 1 else "Genuine"
                        trans_color = "#ff4b4b" if trans_pred_label == 1 else "#4bff64"

                    # --- Display Comparison ---
                    st.markdown("### 🧠 Model Comparison")
                    col1, col2 = st.columns(2)

                    # Ensemble Card
                    with col1:
                        if ensemble_model:
                            st.markdown(f"""
                                <div class='result-box' style='border: 2px solid {ens_color}; background: rgba(0,0,0,0.2);'>
                                    <h3 style='color: #aaa;'>🌲 Ensemble Model</h3>
                                    <h2 style='color: {ens_color}; margin: 10px 0;'>{ens_pred}</h2>
                                    <p style='font-size: 0.9rem;'>Confidence: <b>{ens_conf:.1%}</b></p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("Ensemble model not loaded.")

                    # Transformer Card
                    with col2:
                        if trans_model:
                             st.markdown(f"""
                                <div class='result-box' style='border: 2px solid {trans_color}; background: rgba(0,0,0,0.2);'>
                                    <h3 style='color: #aaa;'>🤖 Transformer (Hugging Face)</h3>
                                    <h2 style='color: {trans_color}; margin: 10px 0;'>{trans_pred}</h2>
                                    <p style='font-size: 0.9rem;'>Confidence: <b>{trans_conf:.1%}</b></p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("Transformer model not loaded.")

                    # History Update
                    final_verdict = trans_pred == "Sarcastic" if trans_model else (ens_pred == "Sarcastic")
                    st.session_state.history.append({"text": user_input, "is_sarcasm": final_verdict})

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
            else:
                st.warning("Please enter some text first!")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        © 2025 Sarcasm AI •
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()