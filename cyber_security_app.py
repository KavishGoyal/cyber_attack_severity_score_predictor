# security_score_app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer with caching
@st.cache_resource
def load_security_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "./models",
        num_labels=1,
        problem_type="regression"
    )
    tokenizer = AutoTokenizer.from_pretrained("./models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

def predict_score(text, model, tokenizer, device):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.logits.squeeze().item()

# Streamlit UI
st.set_page_config(page_title="Cyber Threat Score Predictor", page_icon="🛡️")

# Main content
st.title("🛡️ Cyber Threat Security Score Predictor")
st.markdown("Predict security scores based on CVSS metrics")

# Load model once
model, tokenizer, device = load_security_model()

# Input section
with st.form("prediction_form"):
    cvss_input = st.text_area(
        "Enter CVSS Metrics Vector",
        placeholder="Example: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        height=100
    )
    
    submitted = st.form_submit_button("Predict Security Score")

# Handle prediction
if submitted:
    if not cvss_input.strip():
        st.warning("Please enter CVSS metrics to analyze")
    else:
        try:
            with st.spinner("Analyzing security metrics..."):
                score = predict_score(cvss_input, model, tokenizer, device)
                
            st.success(f"Predicted Security Score: **{score:.2f}**")
            
            # Interpretation guide
            st.markdown("""
            **Score Interpretation:**
            - 0.0-3.9: Low risk
            - 4.0-6.9: Medium risk
            - 7.0-8.9: High risk
            - 9.0-10.0: Critical risk
            """)
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Examples expander
with st.expander("📚 Example CVSS Vectors"):
    st.markdown("""
    **Common CVSS V3.1 Vectors:**
    
    - Critical Risk Example:
    `CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H`  
    Base Score: 9.8
    
    - High Risk Example:
    `CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:L/A:N`  
    Base Score: 7.3
    
    - Medium Risk Example:
    `CVSS:3.1/AV:L/AC:L/PR:H/UI:N/S:U/C:H/I:N/A:N`  
    Base Score: 5.5
    
    - Low Risk Example:
    `CVSS:3.1/AV:L/AC:H/PR:H/UI:R/S:U/C:L/I:N/A:N`  
    Base Score: 2.5
    """)

# Sidebar information
st.sidebar.header("About")
st.sidebar.markdown("""
This AI-powered tool predicts cyber security threat scores based on 
[CVSS v3.1](https://www.first.org/cvss/v3.1/specification-document) metrics vectors.

**How to use:**
1. Paste CVSS vector in the main input
2. Click 'Predict Security Score'
3. Review results and interpretation

**Model Info:**
- Fine-tuned DistilBERT model
- Trained on WebAttack-CVSSMetrics dataset
- Output range: 0.0-10.0
""")