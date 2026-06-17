import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from train_test import OptimizedMedicalImageClassifier

# Initialize Streamlit app
st.title("Medical Image Classification Results Dashboard")

# Paths for results
results_path = './results'
data_path = './features/combined_features.csv'  # Add the correct path to your dataset

# Load saved results
if os.path.exists(results_path):
    # Load metrics
    if os.path.exists(f'{results_path}/metrics.json'):
        with open(f'{results_path}/metrics.json', 'r') as f:
            metrics = json.load(f)
        st.success("Metrics loaded successfully!")
    else:
        metrics = None
        st.warning("Metrics file not found.")

    # Load training history
    if os.path.exists(f'{results_path}/training_history.json'):
        with open(f'{results_path}/training_history.json', 'r') as f:
            training_history = json.load(f)
        st.success("Training history loaded successfully!")
    else:
        training_history = None
        st.warning("Training history file not found.")

    # Load models, scalers, and label encoders
    classifier = OptimizedMedicalImageClassifier(data_path=data_path, results_path=results_path)
    classifier.load_models()
    st.success("Models, scalers, and label encoders loaded successfully!")
else:
    st.error("Results path does not exist. Please train the models first.")

# Display Training History
st.header("Training History")
if training_history:
    for target, history_data in training_history.items():
        with st.expander(f"Training History for {target}"):
            # Plot Loss
            fig, ax = plt.subplots(figsize=(10, 5))
            for fold_idx, fold_history in enumerate(history_data['fold_histories']):
                ax.plot(fold_history['loss'], label=f'Fold {fold_idx + 1} Train', alpha=0.7)
                ax.plot(fold_history['val_loss'], label=f'Fold {fold_idx + 1} Val', linestyle='--', alpha=0.7)
            ax.set_title(f'{target} - Model Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)

            # Plot Accuracy
            fig, ax = plt.subplots(figsize=(10, 5))
            for fold_idx, fold_history in enumerate(history_data['fold_histories']):
                ax.plot(fold_history['accuracy'], label=f'Fold {fold_idx + 1} Train', alpha=0.7)
                ax.plot(fold_history['val_accuracy'], label=f'Fold {fold_idx + 1} Val', linestyle='--', alpha=0.7)
            ax.set_title(f'{target} - Model Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            st.pyplot(fig)
else:
    st.warning("No training history found. Please train the models first.")

# Display Evaluation Metrics
st.header("Evaluation Metrics")
if metrics:
    for target, metric in metrics.items():
        with st.expander(f"Evaluation Metrics for {target}"):
            st.write(f"Accuracy: {metric['accuracy']:.4f}")
            st.write(f"Precision: {metric['precision']:.4f}")
            st.write(f"Recall: {metric['recall']:.4f}")
            st.write(f"F1 Score: {metric['f1']:.4f}")
            st.write("ROC AUC Scores:")
            for class_idx, auc_score in metric['roc_auc'].items():
                st.write(f"Class {metric['class_names'][int(class_idx)]}: {auc_score:.4f}")

            # Confusion Matrix
            st.subheader(f"Confusion Matrix for {target}")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                metric['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=metric['class_names'],
                yticklabels=metric['class_names']
            )
            ax.set_title(f'{target} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
else:
    st.warning("No evaluation metrics found. Please evaluate the models first.")

# Generate Synthetic Test Cases
st.header("Generate Synthetic Test Cases")
n_samples_per_class = st.number_input(
    "Number of samples to generate per class",
    min_value=1,
    max_value=100,
    value=10
)

# Generate Synthetic Test Cases
if st.button("Generate Synthetic Cases"):
    # Load data splits for SMOTE
    data_splits = classifier.load_and_preprocess_data()

    # Generate synthetic cases
    X_synthetic, y_synthetic = classifier.generate_new_cases(data_splits, n_samples_per_class=n_samples_per_class)

    # Create a DataFrame for the synthetic cases (only features, no labels)
    synthetic_df = pd.DataFrame(X_synthetic, columns=[f"feature_{i}" for i in range(X_synthetic.shape[1])])

    # Display synthetic cases
    st.subheader("Generated Synthetic Cases")
    st.write(f"Generated {len(synthetic_df)} samples.")
    st.write(synthetic_df)

    # Download synthetic cases as CSV (only features)
    csv = synthetic_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Synthetic Cases as CSV",
        data=csv,
        file_name="synthetic_cases.csv",
        mime="text/csv"
    )

    # Predict on synthetic cases
    st.subheader("Predictions for Synthetic Cases")
    predictions = classifier.predict_new_case(X_synthetic)

    for i, sample_predictions in enumerate(predictions['target_0']):  # Use 'target_0' as reference
        with st.expander(f"Sample {i + 1}"):
            for target, result in predictions.items():
                st.write(f"**Target:** {target}")
                st.write(f"**Predicted Class:** {result[i]['predicted_class']}")
                st.write(f"**Certainty:** {result[i]['certainty']:.2f}")
                if 'warning' in result[i]:
                    st.warning(result[i]['warning'])
                st.write("**Class Probabilities:**")
                for class_name, prob in result[i]['probabilities'].items():
                    st.write(f"- {class_name}: {prob:.4f}")

# Predict New Case
st.header("Predict New Case")
uploaded_file = st.file_uploader("Upload a CSV file with features", type=["csv"])
if uploaded_file:
    new_data = pd.read_csv(uploaded_file)

    # Check the number of features
    expected_features = classifier.scalers['target_0'].n_features_in_
    if new_data.shape[1] != expected_features:
        st.error(
            f"Invalid number of features. Expected {expected_features} features, but got {new_data.shape[1]} features. "
            f"Please ensure the CSV file contains only the feature columns."
        )
    else:
        predictions = classifier.predict_new_case(new_data.values)

        st.subheader("Prediction Results")
        for i, sample_predictions in enumerate(predictions['target_0']):  # Use 'target_0' as reference
            with st.expander(f"Sample {i + 1}"):
                for target, result in predictions.items():
                    st.write(f"**Target:** {target}")
                    st.write(f"**Predicted Class:** {result[i]['predicted_class']}")
                    st.write(f"**Certainty:** {result[i]['certainty']:.2f}")
                    if 'warning' in result[i]:
                        st.warning(result[i]['warning'])
                    st.write("**Class Probabilities:**")
                    for class_name, prob in result[i]['probabilities'].items():
                        st.write(f"- {class_name}: {prob:.4f}")