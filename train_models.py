import os
import glob
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 

# --- Configuration ---
DATA_PATH = 'data/'
MODELS_PATH = 'saved_models/'
os.makedirs(MODELS_PATH, exist_ok=True)

# --- Feature Extraction ---
def extract_features(file_path):
    """Extracts MFCC features from an audio file."""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Data Loading and Labeling ---
def load_data():
    """Loads data from RAVDESS dataset and assigns labels."""
    features = []
    # RAVDESS filename format: 03-01-01-01-01-01-01.wav
    # Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
    for folder in os.listdir(DATA_PATH):
        if folder.startswith('Actor_'):
            actor_path = os.path.join(DATA_PATH, folder)
            for file_path in glob.glob(os.path.join(actor_path, '*.wav')):
                file_name = os.path.basename(file_path)
                parts = file_name.split('-')
                
                # Labeling based on filename
                emotion = int(parts[2])
                actor = int(parts[6].split('.')[0])
                
                # Gender: odd actor ID = male, even = female
                gender = 'male' if actor % 2 != 0 else 'female'
                
                # Synthetic Age: Assigning ages to actors for demonstration
                # This is a simplification. In a real project, you'd use a dataset with actual age labels.
                actor_ages = {1: 31, 2: 31, 3: 32, 4: 32, 5: 35, 6: 35, 7: 62, 8: 62,
                              9: 25, 10: 25, 11: 41, 12: 41, 13: 45, 14: 45, 15: 68,
                              16: 68, 17: 28, 18: 28, 19: 29, 20: 29, 21: 55, 22: 55,
                              23: 38, 24: 38}
                age = actor_ages.get(actor, 30) # Default age 30
                
                # Emotion mapping
                emotion_map = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                               5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
                
                # Extract features
                mfccs = extract_features(file_path)
                if mfccs is not None:
                    features.append([mfccs, gender, age, emotion_map[emotion]])

    return pd.DataFrame(features, columns=['feature', 'gender', 'age', 'emotion'])

print("Loading and processing data...")
df = load_data()
print("Data loaded successfully.")
print(df.head())

# --- Model Training ---

# 1. Gender Classification Model
print("\n--- Training Gender Classifier ---")
X_gender = np.array(df['feature'].tolist())
y_gender = np.array(df['gender'].tolist())
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gender, y_gender, test_size=0.2, random_state=42)

gender_model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
gender_model.fit(X_train_g, y_train_g)
y_pred_g = gender_model.predict(X_test_g)
accuracy_g = accuracy_score(y_test_g, y_pred_g)
print(f"Gender Model Accuracy: {accuracy_g * 100:.2f}%")
joblib.dump(gender_model, os.path.join(MODELS_PATH, 'gender_model.pkl'))
print("Gender model saved.")

# 2. Age Regression Model (trained only on male voices)
print("\n--- Training Age Regressor ---")
male_df = df[df.gender == 'male']
X_age = np.array(male_df['feature'].tolist())
y_age = np.array(male_df['age'].tolist())
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_age, y_age, test_size=0.2, random_state=42)

age_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
age_model.fit(X_train_a, y_train_a)
y_pred_a = age_model.predict(X_test_a)
mae_a = mean_absolute_error(y_test_a, y_pred_a)
print(f"Age Model MAE: {mae_a:.2f} years")
joblib.dump(age_model, os.path.join(MODELS_PATH, 'age_model.pkl'))
print("Age model saved.")

# 3. Emotion Classification Model
print("\n--- Training Emotion Classifier ---")
X_emotion = np.array(df['feature'].tolist())
y_emotion = np.array(df['emotion'].tolist())
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_emotion, y_emotion, test_size=0.2, random_state=42)

emotion_model = MLPClassifier(hidden_layer_sizes=(200,), max_iter=1000, random_state=42)
emotion_model.fit(X_train_e, y_train_e)
y_pred_e = emotion_model.predict(X_test_e)
accuracy_e = accuracy_score(y_test_e, y_pred_e)
print(f"Emotion Model Accuracy: {accuracy_e * 100:.2f}%")
joblib.dump(emotion_model, os.path.join(MODELS_PATH, 'emotion_model.pkl'))
print("Emotion model saved.")


# --- NEW: Generate and Save Confusion Matrix ---
print("\n--- Generating Confusion Matrix for Emotion Model ---")
# Save plots under evaluation/plots/
PLOTS_DIR = os.path.join('evaluation', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Calculate the confusion matrix
cm = confusion_matrix(y_test_e, y_pred_e, labels=emotion_model.classes_)

# Create a plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=emotion_model.classes_, yticklabels=emotion_model.classes_)
plt.title('Confusion Matrix for Emotion Model')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Save the plot
conf_path = os.path.join(PLOTS_DIR, 'emotion_confusion_matrix.png')
plt.savefig(conf_path)
print(f"Confusion matrix saved to {conf_path}")
# plt.show() # Uncomment this line if you want to see the plot immediately after training

print("\nAll models trained and saved successfully!")