import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)


MODELS_PATH = os.path.join(os.path.dirname(__file__), 'saved_models')
PLOTS_PATH = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_PATH, exist_ok=True)


def load_test_data(features, labels):
    """Utility to ensure numpy arrays for features and labels."""
    X = np.array(features.tolist()) if isinstance(features, pd.Series) else np.array(features)
    y = np.array(labels.tolist()) if isinstance(labels, pd.Series) else np.array(labels)
    return X, y


def evaluate_classifier(model, X_test, y_test, label_prefix):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"{label_prefix} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Confusion matrix plot
    labels = getattr(model, 'classes_', None)
    if labels is None:
        # Try infer from y_test
        labels = np.unique(np.concatenate([y_test, y_pred]))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix: {label_prefix}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plot_path = os.path.join(PLOTS_PATH, f'{label_prefix}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Saved confusion matrix to {plot_path}')

    # Print classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix_path': plot_path,
    }


def evaluate_models_from_files(X_test_gender, y_test_gender,
                               X_test_age, y_test_age,
                               X_test_emotion, y_test_emotion):
    """Loads models from disk and evaluates them against provided test splits.

    Inputs are numpy arrays or pandas Series matching how `train_models.py`
    produces test splits. Returns a dict of metrics and paths to saved plots.
    """
    results = {}

    # Load models
    try:
        gender_model = joblib.load(os.path.join(MODELS_PATH, 'gender_model.pkl'))
    except Exception as e:
        raise FileNotFoundError(f'Could not load gender model: {e}')

    try:
        age_model = joblib.load(os.path.join(MODELS_PATH, 'age_model.pkl'))
    except Exception as e:
        raise FileNotFoundError(f'Could not load age model: {e}')

    try:
        emotion_model = joblib.load(os.path.join(MODELS_PATH, 'emotion_model.pkl'))
    except Exception as e:
        raise FileNotFoundError(f'Could not load emotion model: {e}')

    # Ensure arrays
    Xg, yg = load_test_data(X_test_gender, y_test_gender)
    Xe, ye = load_test_data(X_test_emotion, y_test_emotion)
    Xa, ya = load_test_data(X_test_age, y_test_age)

    # Evaluate classifiers
    results['gender'] = evaluate_classifier(gender_model, Xg, yg, 'gender')
    results['emotion'] = evaluate_classifier(emotion_model, Xe, ye, 'emotion')

    # Age regressor: compute MAE and save scatter plot of predicted vs actual
    y_pred_age = age_model.predict(Xa)
    mae = np.mean(np.abs(ya - y_pred_age))
    results['age'] = {'mae': float(mae)}
    print(f'Age Model MAE: {mae:.4f}')

    plt.figure(figsize=(8, 6))
    plt.scatter(ya, y_pred_age, alpha=0.6)
    plt.plot([min(ya.min(), y_pred_age.min()), max(ya.max(), y_pred_age.max())],
             [min(ya.min(), y_pred_age.min()), max(ya.max(), y_pred_age.max())],
             color='red', linestyle='--')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('Age: Actual vs Predicted')
    age_plot = os.path.join(PLOTS_PATH, 'age_actual_vs_predicted.png')
    plt.savefig(age_plot)
    plt.close()
    results['age']['plot'] = age_plot
    print(f'Saved age scatter plot to {age_plot}')

    return results


if __name__ == '__main__':
    print('This module provides evaluate_models_from_files(X_test_gender, y_test_gender, X_test_age, y_test_age, X_test_emotion, y_test_emotion)')
    print('Import the function in your training script or an interactive session to run evaluations.')
