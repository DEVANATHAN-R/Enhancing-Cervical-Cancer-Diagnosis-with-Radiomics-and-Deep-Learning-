import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from imblearn.over_sampling import SMOTE
import joblib

class OptimizedMedicalImageClassifier:
    def __init__(self, data_path, results_path):
        self.data_path = data_path
        self.results_path = results_path
        self.models = {}
        self.histories = {}
        self.scalers = {}
        self.label_encoders = {}
        self.metrics = {}
        self.num_classes = {}
        self.k_folds = 10  # Number of folds for cross-validation

        if not os.path.exists(results_path):
            os.makedirs(results_path)

    def load_and_preprocess_data(self):
        """Load and preprocess the combined features dataset with data analysis"""
        # Load data
        df = pd.read_csv(self.data_path)

        # Analyze and print dataset information
        print("\nDataset Information:")
        print(f"Total samples: {len(df)}")

        # Separate features by type
        cnn_features = [col for col in df.columns if col.startswith('cnn_feature_')]
        radiomic_features = [col for col in df.columns if col.startswith('radiomic_feature_')]
        metadata_features = [col for col in df.columns if col.startswith('metadata_')]
        target_cols = [col for col in df.columns if col.startswith('target_')]

        print(f"\nFeature Distribution:")
        print(f"CNN features: {len(cnn_features)}")
        print(f"Radiomic features: {len(radiomic_features)}")
        print(f"Metadata features: {len(metadata_features)}")
        print(f"Target variables: {len(target_cols)}")

        # Combine all features
        feature_cols = cnn_features + radiomic_features + metadata_features
        X = df[feature_cols].values
        y_raw = df[target_cols].values

        # Process each target with stratified k-fold cross-validation
        splits = []
        for i in range(y_raw.shape[1]):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_raw[:, i])
            self.label_encoders[f'target_{i}'] = le

            # Analyze class distribution
            unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
            print(f"\nClass distribution for target_{i}:")
            for class_idx, count in zip(unique_classes, class_counts):
                print(f"Class {le.inverse_transform([class_idx])[0]}: {count} samples")

            self.num_classes[f'target_{i}'] = len(unique_classes)
            y_categorical = to_categorical(y_encoded)

            # Create stratified k-fold splits
            skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
            fold_splits = []

            for train_idx, test_idx in skf.split(X, y_encoded):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                fold_splits.append((
                    (X_train_scaled, y_train),
                    (X_test_scaled, y_test)
                ))

            splits.append(fold_splits)
            self.scalers[f'target_{i}'] = scaler

        self.feature_dims = X.shape[1]
        return splits

    def build_model(self, target):
        """Build an optimized neural network model"""
        model = Sequential([
            Dense(512, activation='relu', input_dim=self.feature_dims,
                  kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(self.num_classes[target], activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_models(self, data_splits, epochs=1000, batch_size=32):
        """Train models using k-fold cross-validation"""
        for target_idx, target_splits in enumerate(data_splits):
            target = f'target_{target_idx}'
            print(f"\nTraining model for {target}")

            # Initialize callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
            ]

            # Train on each fold
            fold_histories = []
            fold_models = []

            for fold_idx, ((X_train, y_train), (X_val, y_val)) in enumerate(target_splits):
                print(f"\nTraining fold {fold_idx + 1}/{self.k_folds}")

                model = self.build_model(target)
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )

                fold_histories.append(history.history)
                fold_models.append(model)

            # Select best model based on validation accuracy
            best_model_idx = np.argmax([
                np.max(history['val_accuracy'])
                for history in fold_histories
            ])

            self.models[target] = fold_models[best_model_idx]
            self.histories[target] = {
                'fold_histories': fold_histories,
                'best_fold': best_model_idx
            }

    def save_models(self):
            """Save trained models to disk."""
            for target, model in self.models.items():
                model.save(f'{self.results_path}/model_{target}.h5')

            # Save scalers and label encoders
            joblib.dump(self.scalers, f"{self.results_path}/scalers.pkl")
            joblib.dump(self.label_encoders, f"{self.results_path}/label_encoders.pkl")

    def load_models(self):
        """Load trained models from disk."""
        model_files = [f for f in os.listdir(self.results_path) if f.startswith("model_") and f.endswith(".h5")]
        for model_file in model_files:
            target = model_file.replace("model_", "").replace(".h5", "")
            self.models[target] = tf.keras.models.load_model(os.path.join(self.results_path, model_file))

        # Load scalers and label encoders
        self.scalers = joblib.load(f"{self.results_path}/scalers.pkl")
        self.label_encoders = joblib.load(f"{self.results_path}/label_encoders.pkl")

        print(f"Loaded {len(self.models)} models successfully.")


    def evaluate_models(self, data_splits):
      """Evaluate models using k-fold validation results and calculate weighted average accuracy"""
      total_samples = 0
      weighted_accuracy_sum = 0

      for target_idx, target_splits in enumerate(data_splits):
          target = f'target_{target_idx}'  # Fixed: using target_idx instead of i

          # Initialize arrays to store predictions and true values
          all_predictions = []
          all_true_values = []
          all_probabilities = []

          # Evaluate each fold
          for (_, _), (X_test, y_test) in target_splits:
              # Get predictions
              y_pred_prob = self.models[target].predict(X_test)
              y_pred = np.argmax(y_pred_prob, axis=1)
              y_test_classes = np.argmax(y_test, axis=1)

              # Store results
              all_predictions.extend(y_pred)
              all_true_values.extend(y_test_classes)
              all_probabilities.extend(y_pred_prob)

          # Convert lists to arrays
          all_predictions = np.array(all_predictions)
          all_true_values = np.array(all_true_values)
          all_probabilities = np.array(all_probabilities)

          # Get class names
          le = self.label_encoders[target]
          class_names = le.classes_

          # Calculate metrics
          accuracy = accuracy_score(all_true_values, all_predictions)
          precision, recall, f1, _ = precision_recall_fscore_support(
              all_true_values,
              all_predictions,
              average='weighted'
          )

          # Calculate ROC curve and AUC for each class
          fpr = dict()
          tpr = dict()
          roc_auc = dict()

          n_classes = self.num_classes[target]

          # Create one-hot encoded version of true values for ROC calculation
          y_test_onehot = to_categorical(all_true_values, num_classes=n_classes)

          for class_idx in range(n_classes):
              fpr[class_idx], tpr[class_idx], _ = roc_curve(
                  y_test_onehot[:, class_idx],
                  all_probabilities[:, class_idx]
              )
              roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

          # Store metrics
          self.metrics[target] = {
              'accuracy': accuracy,
              'precision': precision,
              'recall': recall,
              'f1': f1,
              'fpr': fpr,
              'tpr': tpr,
              'roc_auc': roc_auc,
              'confusion_matrix': confusion_matrix(all_true_values, all_predictions),
              'class_names': class_names
          }

          # Print evaluation results
          print(f"\nEvaluation Results for {target}:")
          print(f"Accuracy: {accuracy:.4f}")
          print(f"Precision: {precision:.4f}")
          print(f"Recall: {recall:.4f}")
          print(f"F1 Score: {f1:.4f}")
          print("\nROC AUC Scores:")
          for class_idx in range(n_classes):
              print(f"Class {class_names[class_idx]}: {roc_auc[class_idx]:.4f}")

          # Calculate weighted accuracy contribution
          num_samples = len(all_true_values)
          weighted_accuracy_sum += accuracy * num_samples
          total_samples += num_samples

      # Calculate weighted average accuracy
      weighted_avg_accuracy = weighted_accuracy_sum / total_samples
      print(f"\nOverall Weighted Average Accuracy: {weighted_avg_accuracy:.4f}")

    def save_results(self):
        """Save all results (metrics, plots, etc.) to disk."""
        import json

        # Convert metrics to JSON-serializable format
        serializable_metrics = {}
        for target, metric in self.metrics.items():
            serializable_metrics[target] = {
                'accuracy': float(metric['accuracy']),
                'precision': float(metric['precision']),
                'recall': float(metric['recall']),
                'f1': float(metric['f1']),
                'fpr': {k: v.tolist() for k, v in metric['fpr'].items()},  # Convert ndarray to list
                'tpr': {k: v.tolist() for k, v in metric['tpr'].items()},  # Convert ndarray to list
                'roc_auc': {k: float(v) for k, v in metric['roc_auc'].items()},  # Convert to float
                'confusion_matrix': metric['confusion_matrix'].tolist(),  # Convert ndarray to list
                'class_names': metric['class_names'].tolist() if isinstance(metric['class_names'], np.ndarray) else metric['class_names']  # Handle class_names
            }

        # Save metrics
        with open(f'{self.results_path}/metrics.json', 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

        # Convert training history to JSON-serializable format
        serializable_histories = {}
        for target, history_data in self.histories.items():
            serializable_histories[target] = {
                'best_fold': int(history_data['best_fold']),  # Convert to int
                'fold_histories': [
                    {
                        'loss': [float(x) for x in fold_history['loss']],  # Convert to list of floats
                        'accuracy': [float(x) for x in fold_history['accuracy']],  # Convert to list of floats
                        'val_loss': [float(x) for x in fold_history['val_loss']],  # Convert to list of floats
                        'val_accuracy': [float(x) for x in fold_history['val_accuracy']]  # Convert to list of floats
                    }
                    for fold_history in history_data['fold_histories']
                ]
            }

        # Save training history
        with open(f'{self.results_path}/training_history.json', 'w') as f:
            json.dump(serializable_histories, f, indent=4)

        # Save visualizations
        self.visualize_results()

        print("All results saved successfully.")

    def generate_new_cases(self, data_splits, n_samples_per_class=10):
        """
        Generate new synthetic test cases using SMOTE for all targets.
        Ensure balanced class distribution across all targets.
        :param data_splits: The data splits from load_and_preprocess_data
        :param n_samples_per_class: Number of samples to generate per class
        :return: Synthetic features and labels for all targets
        """
        synthetic_features = []
        synthetic_labels = []

        # Use the first target's feature dimensions as reference
        reference_target = f'target_0'
        reference_scaler = self.scalers[reference_target]

        # Combine all folds for all targets
        X_all = []
        y_all = []

        for target_idx in range(len(data_splits)):
            target = f'target_{target_idx}'
            le = self.label_encoders[target]

            # Combine all folds for the target
            for (X_train, y_train), (X_val, y_val) in data_splits[target_idx]:
                X_all.append(X_train)
                X_all.append(X_val)
                y_all.append(np.argmax(y_train, axis=1))  # Convert one-hot to class indices
                y_all.append(np.argmax(y_val, axis=1))    # Convert one-hot to class indices

        X_all = np.vstack(X_all)
        y_all = np.hstack(y_all)  # Use hstack instead of vstack for class indices

        # Apply SMOTE to generate synthetic samples
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_synthetic, y_synthetic = smote.fit_resample(X_all, y_all)

        # Scale the synthetic features using the reference scaler
        X_synthetic_scaled = reference_scaler.transform(X_synthetic)

        # Select a balanced subset of synthetic samples
        unique_classes, class_counts = np.unique(y_synthetic, return_counts=True)
        synthetic_indices = []

        for class_idx in unique_classes:
            class_samples = np.where(y_synthetic == class_idx)[0]
            selected_indices = np.random.choice(class_samples, size=n_samples_per_class, replace=False)
            synthetic_indices.extend(selected_indices)

        X_synthetic_selected = X_synthetic_scaled[synthetic_indices]
        y_synthetic_selected = y_synthetic[synthetic_indices]

        return X_synthetic_selected, y_synthetic_selected

    def predict_new_case(self, features, return_confidence=True):
        """Make predictions with better confidence handling and testing"""
        predictions = {}

        for target, model in self.models.items():
            # Check if the number of features matches the expected shape
            expected_features = self.scalers[target].n_features_in_
            if features.shape[1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, but got {features.shape[1]} features."
                )

            # Scale features
            features_scaled = self.scalers[target].transform(features)

            # Get predictions and confidence
            pred_prob = model.predict(features_scaled)
            pred_class = np.argmax(pred_prob, axis=1)

            # Get actual class names
            le = self.label_encoders[target]
            class_names = le.classes_

            # Calculate prediction entropy (uncertainty measure)
            entropy = -np.sum(pred_prob * np.log2(pred_prob + 1e-10), axis=1)
            max_entropy = -np.log2(1 / len(class_names))  # maximum possible entropy
            certainty = 1 - (entropy / max_entropy)

            # Store predictions for each sample
            sample_predictions = []
            for i in range(len(features)):
                sample_pred = {
                    'predicted_class': class_names[pred_class[i]],
                    'certainty': float(certainty[i]),
                    'probabilities': {
                        str(class_names[j]): float(pred_prob[i][j])
                        for j in range(len(class_names))
                    },
                    'entropy': float(entropy[i])
                }

                # Add warning if prediction is too certain
                if certainty[i] > 0.95:
                    sample_pred['warning'] = "High certainty prediction - consider validating"

                sample_predictions.append(sample_pred)

            predictions[target] = sample_predictions

        return predictions

    def _plot_training_history(self, timestamp):
      """Plot training history for all models with k-fold results"""
      for target, history_data in self.histories.items():
          plt.figure(figsize=(15, 5))

          # Get histories from all folds
          fold_histories = history_data['fold_histories']
          best_fold = history_data['best_fold']

          # Plot Loss
          plt.subplot(1, 2, 1)
          for fold_idx, fold_history in enumerate(fold_histories):
              alpha = 1.0 if fold_idx == best_fold else 0.3
              plt.plot(fold_history['loss'],
                      label=f'Fold {fold_idx+1} Train',
                      alpha=alpha)
              plt.plot(fold_history['val_loss'],
                      label=f'Fold {fold_idx+1} Val',
                      linestyle='--',
                      alpha=alpha)

          plt.title(f'{target} - Model Loss')
          plt.xlabel('Epoch')
          plt.ylabel('Loss')
          plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

          # Plot Accuracy
          plt.subplot(1, 2, 2)
          for fold_idx, fold_history in enumerate(fold_histories):
              alpha = 1.0 if fold_idx == best_fold else 0.3
              plt.plot(fold_history['accuracy'],
                      label=f'Fold {fold_idx+1} Train',
                      alpha=alpha)
              plt.plot(fold_history['val_accuracy'],
                      label=f'Fold {fold_idx+1} Val',
                      linestyle='--',
                      alpha=alpha)

          plt.title(f'{target} - Model Accuracy')
          plt.xlabel('Epoch')
          plt.ylabel('Accuracy')
          plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

          plt.tight_layout()
          plt.savefig(f'{self.results_path}/training_history_{target}_{timestamp}.png',
                    bbox_inches='tight')
          plt.close()

    def visualize_results(self):
        """Generate and save visualization plots"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create separate plots for each target
        for target in self.models.keys():
            self._plot_training_history(timestamp)
            self._plot_roc_curves(timestamp)
            self._plot_confusion_matrices(timestamp)

        # Create overall performance summary
        self._plot_performance_summary(timestamp)

    def _plot_roc_curves(self, timestamp):
        """Plot ROC curves for all models and classes"""
        for target, metric in self.metrics.items():
            plt.figure(figsize=(10, 8))

            class_names = metric['class_names']
            n_classes = len(class_names)

            for class_idx in range(n_classes):
                plt.plot(
                    metric['fpr'][class_idx],
                    metric['tpr'][class_idx],
                    label=f'{class_names[class_idx]} (AUC = {metric["roc_auc"][class_idx]:.2f})'
                )

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {target}')
            plt.legend(loc="lower right")

            plt.savefig(f'{self.results_path}/roc_curves_{target}_{timestamp}.png')
            plt.close()

    def _plot_confusion_matrices(self, timestamp):
        """Plot confusion matrices for all models"""
        for target, metric in self.metrics.items():
            plt.figure(figsize=(8, 6))

            sns.heatmap(
                metric['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=metric['class_names'],
                yticklabels=metric['class_names']
            )

            plt.title(f'{target} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            plt.tight_layout()
            plt.savefig(f'{self.results_path}/confusion_matrix_{target}_{timestamp}.png')
            plt.close()

    def _plot_performance_summary(self, timestamp):
        """Plot performance metrics summary"""
        metrics_df = pd.DataFrame({
            target: {
                'Accuracy': metric['accuracy'],
                'Precision': metric['precision'],
                'Recall': metric['recall'],
                'F1 Score': metric['f1'],
                'Mean AUC': np.mean(list(metric['roc_auc'].values()))
            }
            for target, metric in self.metrics.items()
        }).T

        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar', width=0.8)
        plt.title('Performance Metrics Summary')
        plt.xlabel('Target')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(f'{self.results_path}/performance_summary_{timestamp}.png',
                    bbox_inches='tight')
        plt.close()

def main():
    # Initialize paths
    data_path = './features/combined_features.csv'
    results_path = './results'

    # Initialize classifier
    classifier = OptimizedMedicalImageClassifier(data_path, results_path)

    # Load and preprocess data
    data_splits = classifier.load_and_preprocess_data()

    # Train models
    classifier.train_models(data_splits)

    # Evaluate models
    classifier.evaluate_models(data_splits)
# Save results (metrics, plots, etc.)
    classifier.save_results()
    # Visualize results
    classifier.visualize_results()

    # Save models
    classifier.save_models()

if __name__ == "__main__":
    main()