import streamlit as st

##HEALTH CHECK
health_param = st.query_params.get("health")
if health_param == "1" or (isinstance(health_param, list) and health_param[0] == "1"):
    st.write("Still alive!🥱")
    st.stop()






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
            model.eval() #set to eval mode
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

#add src to path to import utils properly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    import utils
    #register 'utils' in sys.modules so pickle finds it
    sys.modules['utils'] = utils
except ImportError:
    st.error("Could not import 'src/utils.py'. Please ensure the file exists.")

#streamlit Page Configuration
st.set_page_config(
    page_title="Sarcasm Detector Model",
    page_icon="😏",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'history' not in st.session_state:
    st.session_state.history = []


#custom CSS for modern UI
st.markdown("""

<style>
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%);
        color: #fafafa;
    }
    
    /* Modern Card Container */
    .css-1r6slb0, .stVerticalBlock > div {
        border-radius: 12px;
    }

    /* Input Area Styling */
    .stTextArea textarea {
        background-color: rgba(38, 39, 48, 0.7) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea:focus {
        border: 1px solid #ff4b4b !important;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.2) !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #ff4b4b, #ff3333) !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        transition: transform 0.2s !important;
    }
    .stButton button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4) !important;
    }

    /* Results Card */
    .result-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
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
        #load from Hugging Face instead of local folder
        model_name = "gnetozela/sarcasm_detection"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()  #eval mode
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading transformer from Hugging Face: {e}")
        return None, None

def main():
    #load models on startup
    with st.spinner("Loading AI models..."):
        ensemble_model = load_sarcasm_model()
        trans_tokenizer, trans_model = load_transformer_model()
    
    # Main Content
    st.markdown("<h1 class='stTitle'>is it <span style='color:#ff4b4b'>Sarcastic?</span> </h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Comparing Ensemble Learning vs. Transformer</p>", unsafe_allow_html=True)

    # Layout
    col_main, col_dummy = st.columns([2, 0.5]) 
    with col_main:
        # Helper to handle input state
        if 'input_val' not in st.session_state:
            st.session_state.input_val = ""

        # Quick Examples
        st.markdown("Try an example:")
        examples = [
            "Oh great, another meeting.", 
            "I absolutely love being stuck in traffic.", 
            "The weather is lovely today."
        ]
        
        cols = st.columns(len(examples))
        
       
        input_area_placeholder = st.empty()

        # When an example is clicked
        for i, ex in enumerate(examples):
            if cols[i].button(ex, key=f"ex_{i}"):

                # streaming/typewriter effect directly in the text area
                typed_text = ""
                for char in ex:
                    typed_text += char
                    input_area_placeholder.text_area(
                        "Enter your text:",
                        value=typed_text,
                        placeholder="Type something sarcastic here...",
                        height=120,
                        key=f"input_anim_{i}_{len(typed_text)}" 
                    )
                    time.sleep(0.01) 
                
                st.session_state.input_val = ex
        
        if 'input_val' not in st.session_state:
            st.session_state.input_val = ""

        # Final Render of the Input Area (if not animating)
        user_input = input_area_placeholder.text_area(
            "Enter your text:",
            value=st.session_state.input_val,
            placeholder="Type something sarcastic here...",
            height=120
        )
        
        #update session state if changed
        if user_input != st.session_state.input_val:
            st.session_state.input_val = user_input

        #analyze
        if st.button("🔍 Analyze Tone", use_container_width=True):
            if user_input.strip():
                
                
                # simulated "Thinking" Stream
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

                    # Comparison Display
                    st.markdown("### 🧠 Model Comparison")
                    col1, col2 = st.columns(2)

                    # Ensemble Card
                    with col1:
                        if ensemble_model:
                            st.markdown(f"""
                                <div class='result-box' style='border: 2px solid {ens_color}; background: rgba(0,0,0,0.2);'>
                                    <h3 style='color: #aaa;'>🌲Ensemble Model</h3>
                                    <h2 style='color: {ens_color}; margin: 7px 0;'>{ens_pred}</h2>
                                    <p style='font-size: 0.7rem;'>Confidence: <b>{ens_conf:.1%}</b></p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("Ensemble model not loaded.")

                    # Transformer Card
                    with col2:
                        if trans_model:
                             st.markdown(f"""
                                <div class='result-box' style='border: 2px solid {trans_color}; background: rgba(0,0,0,0.2);'>
                                    <h3 style='color: #aaa;'>🤖Transformer</h3>
                                    <h2 style='color: {trans_color}; margin: 7px 0;'>{trans_pred}</h2>
                                    <p style='font-size: 0.7rem;'>Confidence: <b>{trans_conf:.1%}</b></p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("Transformer model not loaded.")

                    # History Update(Not used in app yet - for future features)
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