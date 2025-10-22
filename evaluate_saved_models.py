"""Script to load data, recreate test splits, and evaluate saved models.

This mirrors the train/test splitting used in `train_models.py` so metrics match the original evaluation.
"""
import os
import glob
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from evaluate_models import evaluate_models_from_files

DATA_PATH = 'data/'
CACHE_PATH = 'cached_features.npz'

# --- Feature Extraction (copied from train_models.py) ---
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Data Loading and Labeling ---
def load_data():
    features = []
    emotion_map = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                   5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    # Synthetic ages mapping (same as in train_models.py)
    actor_ages = {1: 31, 2: 31, 3: 32, 4: 32, 5: 35, 6: 35, 7: 62, 8: 62,
                  9: 25, 10: 25, 11: 41, 12: 41, 13: 45, 14: 45, 15: 68,
                  16: 68, 17: 28, 18: 28, 19: 29, 20: 29, 21: 55, 22: 55,
                  23: 38, 24: 38}

    # Use cached features if available
    if os.path.exists(CACHE_PATH):
        print(f'Loading cached features from {CACHE_PATH}')
        data = np.load(CACHE_PATH, allow_pickle=True)
        feats = data['features']
        genders = data['genders']
        ages = data['ages']
        emotions = data['emotions']
        for f, g, a, e in zip(feats, genders, ages, emotions):
            features.append([f, g, a, e])
        return pd.DataFrame(features, columns=['feature', 'gender', 'age', 'emotion'])

    print('No cache found; extracting features from audio files (this may take several minutes)...')
    total = 0
    for folder in os.listdir(DATA_PATH):
        if folder.startswith('Actor_'):
            actor_path = os.path.join(DATA_PATH, folder)
            total += len(glob.glob(os.path.join(actor_path, '*.wav')))

    processed = 0
    feats_list, genders_list, ages_list, emotions_list = [], [], [], []
    for folder in os.listdir(DATA_PATH):
        if folder.startswith('Actor_'):
            actor_path = os.path.join(DATA_PATH, folder)
            for file_path in glob.glob(os.path.join(actor_path, '*.wav')):
                file_name = os.path.basename(file_path)
                parts = file_name.split('-')

                emotion = int(parts[2])
                actor = int(parts[6].split('.')[0])
                gender = 'male' if actor % 2 != 0 else 'female'
                age = actor_ages.get(actor, 30)

                mfccs = extract_features(file_path)
                if mfccs is not None:
                    features.append([mfccs, gender, age, emotion_map[emotion]])
                    feats_list.append(mfccs)
                    genders_list.append(gender)
                    ages_list.append(age)
                    emotions_list.append(emotion_map[emotion])

                processed += 1
                if processed % 20 == 0 or processed == total:
                    print(f'Extracted features for {processed}/{total} files')

    # Save cache
    try:
        np.savez_compressed(CACHE_PATH, features=feats_list, genders=genders_list, ages=ages_list, emotions=emotions_list)
        print(f'Saved extracted features to cache: {CACHE_PATH}')
    except Exception as e:
        print(f'Could not save feature cache: {e}')

    return pd.DataFrame(features, columns=['feature', 'gender', 'age', 'emotion'])


def main():
    print('Loading data for evaluation...')
    df = load_data()
    print(f'Total samples: {len(df)}')

    # Prepare gender dataset
    X_gender = np.array(df['feature'].tolist())
    y_gender = np.array(df['gender'].tolist())
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gender, y_gender, test_size=0.2, random_state=42)

    # Prepare age dataset (male only)
    male_df = df[df.gender == 'male']
    X_age = np.array(male_df['feature'].tolist())
    y_age = np.array(male_df['age'].tolist())
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_age, y_age, test_size=0.2, random_state=42)

    # Prepare emotion dataset
    X_emotion = np.array(df['feature'].tolist())
    y_emotion = np.array(df['emotion'].tolist())
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_emotion, y_emotion, test_size=0.2, random_state=42)

    # Call evaluator
    print('\nEvaluating saved models...')
    results = evaluate_models_from_files(X_test_g, y_test_g, X_test_a, y_test_a, X_test_e, y_test_e)

    print('\nEvaluation results summary:')
    print(results)

if __name__ == '__main__':
    main()
