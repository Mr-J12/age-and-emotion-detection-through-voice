import streamlit as st
import numpy as np
import librosa
import joblib
import os

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Voice Analysis", layout="wide")

# --- Constants and Model Loading ---
MODELS_PATH = 'saved_models/'

@st.cache_resource
def load_models():
    """Loads all trained models from disk."""
    try:
        gender_model = joblib.load(os.path.join(MODELS_PATH, 'gender_model.pkl'))
        age_model = joblib.load(os.path.join(MODELS_PATH, 'age_model.pkl'))
        emotion_model = joblib.load(os.path.join(MODELS_PATH, 'emotion_model.pkl'))
        return gender_model, age_model, emotion_model
    except FileNotFoundError:
        st.error("Models not found. Please run `train_models.py` first.")
        return None, None, None

gender_model, age_model, emotion_model = load_models()

# --- Feature Extraction Function ---
def extract_features(file):
    """Extracts MFCC features from an uploaded audio file."""
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs.reshape(1, -1)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# --- Streamlit GUI ---
st.title("👨‍🦳 Age and Emotion Detection through Voice 🎤")
st.markdown("Upload a **WAV** audio file of a male speaker to detect their age. If the speaker is a senior citizen (age > 60), their emotion will also be detected.")

# Check if models are loaded before proceeding
if all([gender_model, age_model, emotion_model]):
    uploaded_file = st.file_uploader("Choose a WAV file...", type="wav")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Create a button to trigger analysis
        if st.button("Analyze Voice"):
            with st.spinner('Analyzing...'):
                features = extract_features(uploaded_file)

                if features is not None:
                    # 1. Predict Gender
                    gender_prediction = gender_model.predict(features)[0]

                    # 2. Apply Logic
                    if gender_prediction.lower() != 'male':
                        st.error("Female voice detected. Please upload a male voice.", icon="🚺")
                    else:
                        st.success("Male voice detected.", icon="🚹")

                        # Predict Age for Male Voice
                        age_prediction = age_model.predict(features)[0]
                        age = int(round(age_prediction))
                        st.info(f"**Predicted Age:** {age} years")

                        # Predict Emotion for Senior Citizens
                        if age > 60:
                            st.warning("Senior citizen detected. Analyzing emotion...", icon="👴")
                            emotion_prediction = emotion_model.predict(features)[0]
                            st.success(f"**Predicted Emotion:** {emotion_prediction.capitalize()}")
                        else:
                            st.info("Age is below 60, no emotion analysis required.")
else:
    st.info("Please wait for models to be loaded or run the training script.")