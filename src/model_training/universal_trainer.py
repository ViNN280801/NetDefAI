import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import parallel_backend
from joblib import parallel_backend as joblib_parallel_backend
from src.logger.logger_settings import universal_trainer_logger as logger

# For ANN model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import device manager
from .device_manager import get_device_manager, ComputeDevice


# Define a PyTorch dataset class for text classification
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define a simple ANN model for text classification
class TextClassifierANN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256], dropout_rate=0.5):
        super(TextClassifierANN, self).__init__()

        layers = []
        prev_size = input_size

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)
        return self.sigmoid(x)


class UniversalTrainer:
    """Universal trainer for different types of attack detection models."""

    FEATURE_EXTRACTORS = {
        "count": CountVectorizer,
        "tfidf": TfidfVectorizer,
    }

    MODELS = {
        "logistic": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "mlp": MLPClassifier,
        "ann": "custom_ann",  # Special handler for our PyTorch ANN model
    }

    def __init__(
        self,
        model_type="ann",  # Changed default to ANN
        feature_extractor="tfidf",  # TF-IDF usually works better for text
        model_params=None,
        feature_extractor_params=None,
        preferred_device=None,
        n_jobs=None,
    ):
        """
        Initialize the trainer.

        Args:
            model_type (str): Type of model to use (logistic, random_forest, svm, mlp, ann)
            feature_extractor (str): Type of feature extractor (count, tfidf)
            model_params (dict): Parameters for the model
            feature_extractor_params (dict): Parameters for the feature extractor
            preferred_device: Preferred compute device (from ComputeDevice enum)
            n_jobs (int): Number of jobs for parallel processing (None for auto)
        """
        self.model_type = model_type
        self.feature_extractor_type = feature_extractor
        self.model_params = model_params or {}
        self.feature_extractor_params = feature_extractor_params or {}

        # Setup device configuration
        self.device_manager = get_device_manager()
        self.device = self.device_manager.select_device(preferred_device)

        # Set PyTorch device if using ANN
        if model_type == "ann":
            if self.device == ComputeDevice.CUDA and torch.cuda.is_available():
                self.torch_device = torch.device("cuda")
            elif (
                self.device == ComputeDevice.MPS
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.torch_device = torch.device("mps")
            else:
                self.torch_device = torch.device("cpu")
            logger.info(f"Using PyTorch device: {self.torch_device}")

        # Configure n_jobs for parallel processing
        if n_jobs is None:
            if self.device == ComputeDevice.CPU_MULTI:
                # Use all cores by default
                self.n_jobs = self.device_manager.num_cores
            elif self.device == ComputeDevice.CPU_SERIAL:
                # Serial processing
                self.n_jobs = 1
            else:
                # GPU acceleration, use minimal CPU jobs
                self.n_jobs = 1
        else:
            self.n_jobs = n_jobs

        # Initialize feature extractor
        if feature_extractor not in self.FEATURE_EXTRACTORS:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
        self.feature_extractor = self.FEATURE_EXTRACTORS[feature_extractor](
            **self.feature_extractor_params
        )

        # Initialize model based on type
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_type == "ann":
            # Initialize model after feature extraction
            self.model = None

            # Store ANN specific parameters
            self.ann_params = {
                "hidden_sizes": self.model_params.get("hidden_sizes", [512, 256]),
                "dropout_rate": self.model_params.get("dropout_rate", 0.5),
                "learning_rate": self.model_params.get("learning_rate", 0.001),
                "batch_size": self.model_params.get("batch_size", 64),
                "epochs": self.model_params.get("epochs", 10),
            }
        else:
            # Initialize scikit-learn models immediately
            if "n_jobs" in self.MODELS[model_type]().get_params():
                self.model_params["n_jobs"] = self.n_jobs
                logger.info(
                    f"Setting {model_type} to use {self.n_jobs} CPU cores"
                )
            self.model = self.MODELS[model_type](**self.model_params)

        self.trained = False

        # Print device information
        self.device_manager.print_device_info()

    def preprocess(self, X):
        """
        Preprocess the data based on the feature extractor.

        Args:
            X: Input data (text)

        Returns:
            transformed_X: Transformed features
        """
        if not hasattr(self, "feature_extractor") or self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized")

        if not self.trained:
            # Use appropriate parallelization when available
            if self.device == ComputeDevice.CPU_MULTI:
                with joblib_parallel_backend("threading", n_jobs=self.n_jobs):
                    return self.feature_extractor.fit_transform(X)
            else:
                return self.feature_extractor.fit_transform(X)
        else:
            # Transform is usually faster, but still parallelize when possible
            if self.device == ComputeDevice.CPU_MULTI:
                with joblib_parallel_backend("threading", n_jobs=self.n_jobs):
                    return self.feature_extractor.transform(X)
            else:
                return self.feature_extractor.transform(X)

    def _train_ann(self, X_train, y_train, X_test, y_test):
        """
        Train the ANN model using PyTorch.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            metrics: Training metrics
        """
        # Convert to dense array if needed
        X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
        X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_dense).to(self.torch_device)
        y_train_tensor = (
            torch.FloatTensor(y_train.values).to(self.torch_device).reshape(-1, 1)
        )
        X_test_tensor = torch.FloatTensor(X_test_dense).to(self.torch_device)
        # Not used in evaluation, but we'll keep it for potential future use
        # y_test_tensor = (
        #     torch.FloatTensor(y_test.values).to(self.torch_device).reshape(-1, 1)
        # )

        # Create datasets and data loaders
        train_dataset = TextDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.ann_params["batch_size"], shuffle=True
        )

        # Initialize the model if not already done
        if self.model is None:
            input_size = X_train_dense.shape[1]
            self.model = TextClassifierANN(
                input_size=input_size,
                hidden_sizes=self.ann_params["hidden_sizes"],
                dropout_rate=self.ann_params["dropout_rate"],
            ).to(self.torch_device)

        # Set up loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.ann_params["learning_rate"]
        )

        # Training loop
        self.model.train()
        logger.info(
            f"Training ANN model for {self.ann_params['epochs']} epochs"
        )

        for epoch in range(self.ann_params["epochs"]):
            running_loss = 0.0

            for batch_X, batch_y in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print statistics
            logger.info(
                f"Epoch {epoch+1}/{self.ann_params['epochs']}, Loss: {running_loss/len(train_loader):.4f}"
            )

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            predicted = (outputs >= 0.5).float().cpu().numpy().flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_test, predicted)
        report = classification_report(y_test, predicted, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predicted)

        logger.info(f"ANN model accuracy: {accuracy:.4f}")
        logger.info(
            f"Classification report:\n{classification_report(y_test, predicted)}"
        )

        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": conf_matrix,
            "y_test": y_test,
            "y_pred": predicted,
        }

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model on the given data.

        Args:
            X: Input data (text)
            y: Target labels
            test_size: Size of the test set
            random_state: Random state for reproducibility

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        logger.info(
            f"Training {self.model_type} model with {self.feature_extractor_type} features"
        )

        # Preprocess data
        logger.info("Preprocessing data...")
        X_transformed = self.preprocess(X)

        # Split data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=test_size, random_state=random_state
        )

        # Train model with appropriate approach based on model type
        logger.info("Training model...")

        if self.model_type == "ann":
            # Use custom ANN training function
            metrics = self._train_ann(X_train, y_train, X_test, y_test)
        else:
            # For scikit-learn models
            if self.device == ComputeDevice.CPU_MULTI:
                # Use parallel backend for multi-core CPU
                with parallel_backend("threading", n_jobs=self.n_jobs):
                    self.model.fit(X_train, y_train)
            else:
                # Default serial implementation
                self.model.fit(X_train, y_train)

            # Evaluate model
            logger.info("Evaluating model...")
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            logger.info(f"Model accuracy: {accuracy:.4f}")
            logger.info(
                f"Classification report:\n{classification_report(y_test, y_pred)}"
            )

            metrics = {
                "accuracy": accuracy,
                "report": report,
                "confusion_matrix": conf_matrix,
                "y_test": y_test,
                "y_pred": y_pred,
            }

        self.trained = True
        return metrics

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input data (text)

        Returns:
            predictions: Predicted labels
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        X_transformed = self.preprocess(X)

        if self.model_type == "ann":
            # Convert to dense array if needed
            X_dense = (
                X_transformed.toarray()
                if hasattr(X_transformed, "toarray")
                else X_transformed
            )

            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X_dense).to(self.torch_device)

            # Make predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predictions = (outputs >= 0.5).float().cpu().numpy().flatten()

            return predictions
        else:
            return self.model.predict(X_transformed)

    def predict_proba(self, X):
        """
        Make probability predictions on new data.

        Args:
            X: Input data (text)

        Returns:
            probabilities: Predicted probabilities
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        X_transformed = self.preprocess(X)

        if self.model_type == "ann":
            # Convert to dense array if needed
            X_dense = (
                X_transformed.toarray()
                if hasattr(X_transformed, "toarray")
                else X_transformed
            )

            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X_dense).to(self.torch_device)

            # Make predictions
            self.model.eval()
            with torch.no_grad():
                probs = self.model(X_tensor).cpu().numpy()

            # Return probabilities for both classes (0 and 1)
            return np.hstack((1 - probs, probs))
        else:
            if not hasattr(self.model, "predict_proba"):
                raise ValueError(
                    f"{self.model_type} does not support probability predictions"
                )

            return self.model.predict_proba(X_transformed)

    def tune_hyperparameters(self, X, y, param_grid, cv=5, scoring="accuracy"):
        """
        Tune hyperparameters using grid search.

        Args:
            X: Input data (text)
            y: Target labels
            param_grid: Parameter grid for grid search
            cv: Number of cross-validation folds
            scoring: Scoring metric for grid search

        Returns:
            best_params: Best parameters found
        """
        logger.info("Tuning hyperparameters...")

        X_transformed = self.preprocess(X)

        # Configure GridSearchCV with appropriate parallelization
        if self.device == ComputeDevice.CPU_MULTI:
            n_jobs = self.n_jobs
        else:
            n_jobs = 1

        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1
        )

        # Run grid search with appropriate backend
        if self.device == ComputeDevice.CPU_MULTI:
            with parallel_backend("threading", n_jobs=self.n_jobs):
                grid_search.fit(X_transformed, y)
        else:
            grid_search.fit(X_transformed, y)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.model = self.MODELS[self.model_type](**grid_search.best_params_)

        return grid_search.best_params_

    def cross_validate(self, X, y, cv=5, scoring="accuracy"):
        """
        Perform cross-validation on the model.

        Args:
            X: Input data (text)
            y: Target labels
            cv: Number of cross-validation folds
            scoring: Scoring metric for cross-validation

        Returns:
            scores: Cross-validation scores
        """
        logger.info(
            f"Performing {cv}-fold cross-validation with {scoring} scoring"
        )

        X_transformed = self.preprocess(X)

        # Use appropriate parallelization
        if self.device == ComputeDevice.CPU_MULTI:
            with parallel_backend("threading", n_jobs=self.n_jobs):
                scores = cross_val_score(
                    self.model, X_transformed, y, cv=cv, scoring=scoring
                )
        else:
            scores = cross_val_score(
                self.model, X_transformed, y, cv=cv, scoring=scoring
            )

        logger.info(
            f"Mean {scoring} score: {scores.mean():.4f} (std: {scores.std():.4f})"
        )
        return scores

    def save(self, filepath):
        """
        Save the model and feature extractor to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.trained:
            logger.warning("Saving untrained model")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # For ANN models, save PyTorch model separately
        if self.model_type == "ann":
            model_state = self.model.state_dict()

            # Save PyTorch model state
            torch_path = filepath.replace(".joblib", "_pytorch_model.pt")
            torch.save(model_state, torch_path)
            logger.info(f"PyTorch model saved to {torch_path}")

            # Save the rest of the data without the PyTorch model
            joblib.dump(
                {
                    "model_type": self.model_type,
                    "feature_extractor": self.feature_extractor,
                    "feature_extractor_type": self.feature_extractor_type,
                    "feature_extractor_params": self.feature_extractor_params,
                    "ann_params": self.ann_params,
                    "input_size": next(self.model.parameters()).shape[
                        1
                    ],  # Save input size for model reconstruction
                    "trained": self.trained,
                    "n_jobs": self.n_jobs,
                },
                filepath,
            )
        else:
            # Save scikit-learn model and feature extractor
            joblib.dump(
                {
                    "model": self.model,
                    "feature_extractor": self.feature_extractor,
                    "model_type": self.model_type,
                    "feature_extractor_type": self.feature_extractor_type,
                    "model_params": self.model_params,
                    "feature_extractor_params": self.feature_extractor_params,
                    "trained": self.trained,
                    "n_jobs": self.n_jobs,
                },
                filepath,
            )

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, preferred_device=None):
        """
        Load model from disk.

        Args:
            filepath: Path to load the model from
            preferred_device: Preferred compute device

        Returns:
            trainer: Loaded trainer instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")

        # Load model and feature extractor
        data = joblib.load(filepath)
        model_type = data["model_type"]

        # Check if it's an ANN model
        if model_type == "ann":
            # Create new instance
            trainer = cls(
                model_type=model_type,
                feature_extractor=data["feature_extractor_type"],
                model_params=data.get("ann_params", {}),
                feature_extractor_params=data.get("feature_extractor_params", {}),
                preferred_device=preferred_device,
                n_jobs=data.get("n_jobs", None),
            )

            # Restore feature extractor
            trainer.feature_extractor = data["feature_extractor"]

            # Load PyTorch model
            torch_path = filepath.replace(".joblib", "_pytorch_model.pt")
            if not os.path.exists(torch_path):
                raise FileNotFoundError(f"PyTorch model file {torch_path} not found")

            # Initialize model architecture
            input_size = data.get("input_size")
            trainer.model = TextClassifierANN(
                input_size=input_size,
                hidden_sizes=trainer.ann_params["hidden_sizes"],
                dropout_rate=trainer.ann_params["dropout_rate"],
            ).to(trainer.torch_device)

            # Load model weights
            trainer.model.load_state_dict(
                torch.load(torch_path, map_location=trainer.torch_device)
            )
            trainer.model.eval()  # Set model to evaluation mode

            trainer.trained = data["trained"]
        else:
            # Create new instance for scikit-learn models
            trainer = cls(
                model_type=model_type,
                feature_extractor=data["feature_extractor_type"],
                model_params=data.get("model_params", {}),
                feature_extractor_params=data.get("feature_extractor_params", {}),
                preferred_device=preferred_device,
                n_jobs=data.get("n_jobs", None),
            )

            # Restore model and feature extractor
            trainer.model = data["model"]
            trainer.feature_extractor = data["feature_extractor"]
            trainer.trained = data["trained"]

        logger.info(f"Model loaded from {filepath}")
        return trainer


def menu():
    """Display the interactive menu for the trainer."""
    print("\nMenu:")
    print("1. Create a new model")
    print("2. Train the model")
    print("3. Evaluate the model")
    print("4. Save the model")
    print("5. Load a model")
    print("6. Make predictions")
    print("7. Tune hyperparameters")
    print("8. Exit")


def main():
    """Main function for interactive model training."""
    trainer = None

    while True:
        menu()
        choice = input("Select an action: ")

        if choice == "1":
            model_type = input(
                "Enter model type (logistic, random_forest, svm, mlp, ann): "
            ).lower()
            feature_extractor = input(
                "Enter feature extractor (count, tfidf): "
            ).lower()

            try:
                trainer = UniversalTrainer(
                    model_type=model_type, feature_extractor=feature_extractor
                )
                print(
                    f"Created new {model_type} model with {feature_extractor} feature extraction"
                )
            except ValueError as e:
                print(f"Error: {e}")

        elif choice == "2":
            if trainer is None:
                print("Please create a model first")
                continue

            dataset_path = input("Enter path to dataset (CSV): ")
            if not os.path.exists(dataset_path):
                print("File not found!")
                continue

            try:
                data = pd.read_csv(dataset_path)
                if "query" not in data.columns or "label" not in data.columns:
                    print("Invalid dataset format! Need 'query' and 'label' columns.")
                    continue

                X = data["query"]
                y = data["label"]

                metrics = trainer.train(X, y)
                print(f"Training completed with accuracy: {metrics['accuracy']:.4f}")
            except Exception as e:
                print(f"Error during training: {e}")

        elif choice == "3":
            if trainer is None or not trainer.trained:
                print("Please train the model first")
                continue

            dataset_path = input("Enter path to evaluation dataset (CSV): ")
            if not os.path.exists(dataset_path):
                print("File not found!")
                continue

            try:
                data = pd.read_csv(dataset_path)
                if "query" not in data.columns or "label" not in data.columns:
                    print("Invalid dataset format! Need 'query' and 'label' columns.")
                    continue

                X = data["query"]
                y = data["label"]

                y_pred = trainer.predict(X)
                accuracy = accuracy_score(y, y_pred)
                print(f"Evaluation accuracy: {accuracy:.4f}")
                print(f"Classification report:\n{classification_report(y, y_pred)}")
            except Exception as e:
                print(f"Error during evaluation: {e}")

        elif choice == "4":
            if trainer is None:
                print("Please create a model first")
                continue

            model_path = input("Enter filename to save model (without extension): ")
            try:
                trainer.save(f"{model_path}.joblib")
            except Exception as e:
                print(f"Error saving model: {e}")

        elif choice == "5":
            model_path = input("Enter path to saved model (.joblib): ")
            if not os.path.exists(model_path):
                print("File not found!")
                continue

            try:
                trainer = UniversalTrainer.load(model_path)
                print("Model successfully loaded!")
            except Exception as e:
                print(f"Error loading model: {e}")

        elif choice == "6":
            if trainer is None or not trainer.trained:
                print("Please train the model first")
                continue

            input_text = input("Enter text to predict: ")
            try:
                prediction = trainer.predict([input_text])[0]
                print(f"Prediction: {prediction}")

                if hasattr(trainer.model, "predict_proba"):
                    proba = trainer.predict_proba([input_text])[0]
                    print(f"Probability: {proba}")
            except Exception as e:
                print(f"Error making prediction: {e}")

        elif choice == "7":
            if trainer is None:
                print("Please create a model first")
                continue

            dataset_path = input("Enter path to dataset (CSV): ")
            if not os.path.exists(dataset_path):
                print("File not found!")
                continue

            try:
                data = pd.read_csv(dataset_path)
                if "query" not in data.columns or "label" not in data.columns:
                    print("Invalid dataset format! Need 'query' and 'label' columns.")
                    continue

                X = data["query"]
                y = data["label"]

                print("Enter parameter grid as Python dictionary:")
                param_grid_str = input()

                param_grid = eval(param_grid_str)
                trainer.tune_hyperparameters(X, y, param_grid)
            except Exception as e:
                print(f"Error tuning hyperparameters: {e}")

        elif choice == "8":
            if trainer is not None and trainer.trained:
                save_choice = input(
                    "You have an unsaved model. Save before exit? (y/n): "
                ).lower()
                if save_choice == "y":
                    model_path = input(
                        "Enter filename to save model (without extension): "
                    )
                    try:
                        trainer.save(f"{model_path}.joblib")
                    except Exception as e:
                        print(f"Error saving model: {e}")

            print("Exiting program...")
            sys.exit(0)

        else:
            print("Invalid choice! Try again.")


if __name__ == "__main__":
    main()
