# app.py
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set page configuration
st.set_page_config(
    page_title="CyberGuard - Bullying Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function (must match your training pipeline)
def preprocess_text(text):
    """
    Clean and preprocess the input text
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens 
                     if word not in stop_words and len(word) > 2]
    
    return ' '.join(cleaned_tokens)

# Load model and vectorizer
@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    try:
        model = joblib.load('cyberbullying_classifier_rf.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please make sure 'cyberbullying_classifier_rf.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        return None, None

# Prediction function
def predict_cyberbullying(text, model, vectorizer):
    """Make prediction on input text"""
    cleaned_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)
    prediction_proba = model.predict_proba(text_tfidf)
    
    label = "CYBERBULLYING" if prediction[0] == 1 else "NORMAL"
    confidence = prediction_proba[0][prediction[0]]
    
    return label, confidence, prediction_proba

# Main application
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    .bullying {
        background-color: #ffcccc;
        border-left: 5px solid #ff4b4b;
    }
    .normal {
        background-color: #ccffcc;
        border-left: 5px solid #4caf50;
    }
    .confidence-bar {
        height: 20px;
        background-color: #ddd;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🛡️ CyberGuard</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Cyberbullying Detection System")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This tool uses machine learning to detect cyberbullying in social media text.
        It analyzes patterns and language cues to identify potentially harmful content.
        """)
        
        st.header("How to Use")
        st.write("""
        1. Type or paste text into the input box
        2. Click 'Analyze Text' 
        3. View the results and confidence level
        """)
        
        st.header("Model Information")
        st.write("Powered by Random Forest classifier trained on social media data")
        
        # File upload for batch processing
        st.header("Batch Analysis")
        uploaded_file = st.file_uploader("Upload a CSV file for batch processing", 
                                       type=['csv'],
                                       help="CSV should have a 'text' column")
        if uploaded_file is not None:
            batch_process(uploaded_file)

    # Load models
    model, vectorizer = load_models()
    if model is None or vectorizer is None:
        return

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        st.subheader("🔍 Analyze Text")
        input_text = st.text_area(
            "Enter social media text to analyze:",
            height=150,
            placeholder="Type or paste text here...",
            help="Example: 'You are so stupid and worthless'"
        )
        
        # Analyze button
        if st.button("🚀 Analyze Text", use_container_width=True):
            if input_text.strip():
                with st.spinner("Analyzing text..."):
                    label, confidence, probabilities = predict_cyberbullying(input_text, model, vectorizer)
                    
                    # Display results
                    st.subheader("📊 Results")
                    
                    # Confidence visualization
                    st.write(f"**Confidence:** {confidence:.2%}")
                    
                    # Color-coded prediction box
                    if label == "CYBERBULLYING":
                        st.markdown(f"""
                        <div class="prediction-box bullying">
                            <h3>⚠️ Prediction: {label}</h3>
                            <p>This content appears to contain cyberbullying elements.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box normal">
                            <h3>✅ Prediction: {label}</h3>
                            <p>This content appears to be normal.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    st.subheader("📈 Probability Breakdown")
                    prob_data = {
                        'Category': ['Normal', 'Cyberbullying'],
                        'Probability': [probabilities[0][0], probabilities[0][1]]
                    }
                    prob_df = pd.DataFrame(prob_data)
                    
                    fig, ax = plt.subplots()
                    bars = ax.bar(prob_df['Category'], prob_df['Probability'], 
                                 color=['#4caf50', '#ff4b4b'])
                    ax.set_ylabel('Probability')
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.2%}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    
            else:
                st.warning("Please enter some text to analyze.")

    with col2:
        st.subheader("📊 Statistics")
        
        # Example statistics (you can replace with actual data)
        st.metric("Accuracy", "92%")
        st.metric("Precision", "89%")
        st.metric("Recall", "94%")
        
        st.subheader("🔍 Top Indicators")
        st.write("""
        Common bullying patterns:
        - Personal attacks
        - Hate speech
        - Threats
        - Body shaming
        - Exclusion language
        """)
        
        st.subheader("⚡ Quick Analysis")
        quick_examples = [
            "You're such a loser, nobody likes you",
            "Great game today! Well played everyone!",
            "Your opinion doesn't matter here"
        ]
        
        for example in quick_examples:
            if st.button(f"Analyze: '{example[:20]}...'", key=example):
                label, confidence, _ = predict_cyberbullying(example, model, vectorizer)
                st.write(f"**Result:** {label} ({confidence:.2%})")

# Batch processing function
def batch_process(uploaded_file):
    """Process a CSV file with multiple texts"""
    try:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column")
            return
        
        model, vectorizer = load_models()
        if model is None:
            return
        
        results = []
        with st.spinner("Processing batch file..."):
            for text in df['text']:
                label, confidence, _ = predict_cyberbullying(str(text), model, vectorizer)
                results.append({
                    'text': text,
                    'prediction': label,
                    'confidence': confidence
                })
        
        results_df = pd.DataFrame(results)
        
        st.subheader("📦 Batch Results")
        st.dataframe(results_df)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="cyberbullying_predictions.csv",
            mime="text/csv"
        )
        
        # Show statistics
        bullying_count = len(results_df[results_df['prediction'] == 'CYBERBULLYING'])
        st.metric("Bullying Content Detected", f"{bullying_count}/{len(results_df)}")
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
