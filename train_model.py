#!/usr/bin/env python3
"""
Universal model training tool for web attack detection.

This script trains a unified model for detecting multiple types of web attacks
using the UniversalTrainer with hardware acceleration.
"""

import os
import glob
import time
import psutil
import argparse
import platform
import datetime
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from src.logger.logger_settings import universal_trainer_logger as logger
from src.model_training import UniversalTrainer, ComputeDevice, get_device_manager

# Try to import GPU monitoring tools with fallbacks
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


# Resource tracking class
class ResourceTracker:
    """Tracks system resources like CPU, memory, and GPU usage during training."""

    def __init__(self, track_gpu=False):
        self.track_gpu = track_gpu and (TORCH_AVAILABLE or GPUTIL_AVAILABLE)
        self.start_time = None
        self.end_time = None
        self.cpu_percent = []
        self.memory_percent = []
        self.memory_used = []
        self.gpu_percent = []
        self.gpu_memory = []
        self.proc = psutil.Process()
        self.sample_interval = 0.5  # seconds
        self.is_tracking = False
        self.tracking_thread = None

        # Get system info
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cpu_count = psutil.cpu_count(logical=False)
        self.total_memory = psutil.virtual_memory().total

        # GPU info
        self.gpu_info = None
        if self.track_gpu:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.gpu_info = {
                    "count": torch.cuda.device_count(),
                    "name": [
                        torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())
                    ],
                    "total_memory": [
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count())
                    ],
                }
            elif GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_info = {
                        "count": len(gpus),
                        "name": [gpu.name for gpu in gpus],
                        "total_memory": [
                            gpu.memoryTotal * 1024 * 1024 for gpu in gpus
                        ],  # Convert to bytes
                    }

    def start(self):
        """Start tracking resources."""
        self.start_time = time.time()
        self.is_tracking = True

        # Start a background thread to sample resource usage
        import threading

        self.tracking_thread = threading.Thread(target=self._track_resources)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()

    def stop(self):
        """Stop tracking resources."""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2.0)
        self.end_time = time.time()

    def _track_resources(self):
        """Background thread to sample resource usage."""
        while self.is_tracking:
            # CPU usage (this process only)
            self.cpu_percent.append(
                self.proc.cpu_percent(interval=None) / self.cpu_count
            )

            # Memory usage (this process only)
            mem_info = self.proc.memory_info()
            self.memory_used.append(mem_info.rss)
            self.memory_percent.append(mem_info.rss / self.total_memory * 100)

            # GPU usage
            if self.track_gpu:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        # This is a placeholder as PyTorch doesn't provide direct GPU utilization
                        # In a real implementation, you might use pynvml or other libraries
                        self.gpu_percent.append(0.0)  # Placeholder
                        self.gpu_memory.append(torch.cuda.memory_allocated())
                    except Exception as e:
                        logger.warning(f"Error getting GPU stats: {e}")
                elif GPUTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            self.gpu_percent.append(gpus[0].load * 100)
                            self.gpu_memory.append(
                                gpus[0].memoryUsed * 1024 * 1024
                            )  # Convert to bytes
                    except Exception as e:
                        logger.warning(f"Error getting GPU stats: {e}")

            time.sleep(self.sample_interval)

    def get_resume(self) -> Dict[str, Any]:
        """
        Generate a resume of resource usage during training.

        Returns:
            Dict containing resource usage statistics
        """
        if not self.cpu_percent:
            return {"error": "No data collected"}

        if not self.start_time:
            return {"error": "Training not started, start_time is None"}

        if not self.end_time:
            return {"error": "Training not completed, end_time is None"}

        duration = (
            self.end_time - self.start_time
            if self.end_time
            else time.time() - self.start_time
        )

        resume = {
            "system_info": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": self.cpu_count,
                "physical_cpu_count": self.physical_cpu_count,
                "total_memory": self.total_memory,
                "gpu_info": self.gpu_info,
            },
            "duration": {
                "seconds": duration,
                "formatted": str(datetime.timedelta(seconds=int(duration))),
            },
            "cpu": {
                "min_percent": min(self.cpu_percent) if self.cpu_percent else 0,
                "avg_percent": (
                    sum(self.cpu_percent) / len(self.cpu_percent)
                    if self.cpu_percent
                    else 0
                ),
                "max_percent": max(self.cpu_percent) if self.cpu_percent else 0,
            },
            "memory": {
                "min_used": min(self.memory_used) if self.memory_used else 0,
                "avg_used": (
                    sum(self.memory_used) / len(self.memory_used)
                    if self.memory_used
                    else 0
                ),
                "max_used": max(self.memory_used) if self.memory_used else 0,
                "min_percent": min(self.memory_percent) if self.memory_percent else 0,
                "avg_percent": (
                    sum(self.memory_percent) / len(self.memory_percent)
                    if self.memory_percent
                    else 0
                ),
                "max_percent": max(self.memory_percent) if self.memory_percent else 0,
            },
        }

        if self.track_gpu and (self.gpu_percent or self.gpu_memory):
            resume["gpu"] = {
                "min_percent": min(self.gpu_percent) if self.gpu_percent else 0,
                "avg_percent": (
                    sum(self.gpu_percent) / len(self.gpu_percent)
                    if self.gpu_percent
                    else 0
                ),
                "max_percent": max(self.gpu_percent) if self.gpu_percent else 0,
                "min_memory": min(self.gpu_memory) if self.gpu_memory else 0,
                "avg_memory": (
                    sum(self.gpu_memory) / len(self.gpu_memory)
                    if self.gpu_memory
                    else 0
                ),
                "max_memory": max(self.gpu_memory) if self.gpu_memory else 0,
            }

        return resume

    def print_resume(self):
        """Print a formatted resume of resource usage to the console."""
        resume = self.get_resume()
        if "error" in resume:
            logger.warning(f"Unable to print resume: {resume['error']}")
            return

        logger.info("\n" + "=" * 50)
        logger.info("TRAINING RESUME")
        logger.info("=" * 50)

        # System information
        logger.info(
            "\n[1] SYSTEM INFORMATION\n"
            + f"Platform: {resume['system_info']['platform']}\n"
            + f"Processor: {resume['system_info']['processor']}\n"
            + f"Logical CPU cores: {resume['system_info']['cpu_count']}\n"
            + f"Physical CPU cores: {resume['system_info']['physical_cpu_count']}\n"
            + f"Total memory: {resume['system_info']['total_memory'] / (1024**3):.2f} GB\n"
        )

        if resume["system_info"]["gpu_info"]:
            logger.info(f"\nGPU count: {resume['system_info']['gpu_info']['count']}")
            for i, (name, mem) in enumerate(
                zip(
                    resume["system_info"]["gpu_info"]["name"],
                    resume["system_info"]["gpu_info"]["total_memory"],
                )
            ):
                logger.info(f"GPU {i}: {name}, Memory: {mem / (1024**3):.2f} GB")

        # Duration
        logger.info(
            "\n[2] TIME SPENT\n"
            + f"Total duration: {resume['duration']['formatted']} (HH:MM:SS)\n"
            + f"Total seconds: {resume['duration']['seconds']:.2f} s\n"
        )

        # CPU usage
        logger.info(
            "\n[3] CPU USAGE (this process)\n"
            + f"Minimum CPU load: {resume['cpu']['min_percent']:.2f}%\n"
            + f"Average CPU load: {resume['cpu']['avg_percent']:.2f}%\n"
            + f"Maximum CPU load: {resume['cpu']['max_percent']:.2f}%\n"
        )

        # Memory usage
        logger.info(
            "\n[4] MEMORY USAGE (this process)\n"
            + f"Minimum memory usage: {resume['memory']['min_used'] / (1024**3):.2f} GB"
            + f"({resume['memory']['min_percent']:.2f}%)\n"
            + f"Average memory usage: {resume['memory']['avg_used'] / (1024**3):.2f} GB"
            + f"({resume['memory']['avg_percent']:.2f}%)\n"
            + f"Maximum memory usage: {resume['memory']['max_used'] / (1024**3):.2f} GB"
            + f"({resume['memory']['max_percent']:.2f}%)\n"
        )

        # GPU usage (if available)
        if "gpu" in resume:
            logger.info(
                "\n[5] GPU USAGE\n"
                + f"Minimum GPU load: {resume['gpu']['min_percent']:.2f}%\n"
                + f"Average GPU load: {resume['gpu']['avg_percent']:.2f}%\n"
                + f"Maximum GPU load: {resume['gpu']['max_percent']:.2f}%\n"
                + f"Minimum GPU memory: {resume['gpu']['min_memory'] / (1024**3):.2f} GB\n"
                + f"Average GPU memory: {resume['gpu']['avg_memory'] / (1024**3):.2f} GB\n"
                + f"Maximum GPU memory: {resume['gpu']['max_memory'] / (1024**3):.2f} GB\n"
            )

        logger.info("\n" + "=" * 50)


def load_all_attack_data(data_dir: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Load data from multiple CSV files for all attack types.

    Args:
        data_dir: Directory containing dataset files for different attacks

    Returns:
        X: Input data
        y: Target labels (1 for malicious, 0 for normal)
        attack_types: Series indicating the type of attack for each sample
    """
    # Initialize lists to store data
    all_inputs = []
    all_labels = []
    all_attack_types = []

    # File patterns for attack types
    attack_patterns = {
        "path_traversal": ["path_traversal", "path"],
        "sql_injection": ["sql_injection", "sql"],
        "xss": ["xss"],
        "dos": ["dos"],
    }

    # Define folder mappings (for organized structure)
    attack_folders = {
        "PathTraversal": "path_traversal",
        "SQLInjections": "sql_injection",
        "XSS": "xss",
        "DoS": "dos",
    }

    # Try first to find CSV files directly in the data directory
    direct_csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    # Also look in subdirectories if they exist
    subfolder_csv_files = []
    for attack_folder in attack_folders:
        folder_path = os.path.join(data_dir, attack_folder)
        if os.path.exists(folder_path):
            subfolder_csv_files.extend(
                glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
            )

    # Combine all found CSV files
    csv_files = direct_csv_files + subfolder_csv_files

    if not csv_files:
        raise FileNotFoundError(f"No dataset files found in {data_dir}")

    logger.info(f"Found {len(csv_files)} dataset files")

    # Process each CSV file
    for file_path in csv_files:
        file_name = os.path.basename(file_path).lower()

        # Determine attack type from subdirectory structure first if possible
        attack_type = None
        for folder, attack_name in attack_folders.items():
            if folder in file_path:
                attack_type = attack_name
                break

        # If attack type not determined from folder structure, try filename patterns
        if not attack_type:
            for attack_name, patterns in attack_patterns.items():
                if any(pattern in file_name for pattern in patterns):
                    attack_type = attack_name
                    break

        if not attack_type:
            logger.warning(f"Could not determine attack type for {file_path}, skipping")
            continue

        try:
            # Load the data
            data = pd.read_csv(file_path)

            # Determine the feature column based on attack type
            if attack_type == "path_traversal":
                feature_col = "path"
            elif attack_type == "sql_injection":
                feature_col = "query"
            elif attack_type == "xss":
                feature_col = "payload"
            elif attack_type == "dos":
                feature_col = "request"
            else:
                feature_col = "query"  # Default

            # Check if required columns exist
            if feature_col not in data.columns or "label" not in data.columns:
                logger.warning(
                    f"File {file_path} missing required columns '{feature_col}' or 'label', skipping"
                )
                continue

            # Extract inputs and labels
            inputs = data[feature_col].astype(str)
            labels = data["label"]

            # Add data to lists
            all_inputs.extend(inputs)
            all_labels.extend(labels)
            all_attack_types.extend([attack_type] * len(inputs))

            logger.info(
                f"Loaded {len(inputs)} samples from {os.path.basename(file_path)} ({attack_type})"
            )

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    if not all_inputs:
        raise ValueError("No valid data was loaded from any files")

    # Convert lists to pandas Series
    X = pd.Series(all_inputs)
    y = pd.Series(all_labels)
    attack_types = pd.Series(all_attack_types)

    # Display dataset statistics
    logger.info(
        "Dataset statistics:\n"
        + f"Total samples: {len(X)}\n"
        + f"Malicious samples: {sum(y == 1)}\n"
        + f"normal samples: {sum(y == 0)}\n"
        + f"Path Traversal samples: {sum(attack_types == 'path_traversal')}\n"
        + f"SQL Injection samples: {sum(attack_types == 'sql_injection')}\n"
        + f"XSS samples: {sum(attack_types == 'xss')}\n"
        + f"DoS samples: {sum(attack_types == 'dos')}"
    )

    return X, y, attack_types


def get_unified_feature_extractor_params() -> Dict[str, Any]:
    """
    Get feature extractor parameters for a unified model.

    Returns:
        params: Feature extractor parameters
    """
    # These parameters are designed to work well across all attack types
    params = {
        "analyzer": "char",  # Character-level analysis works well for all attack types
        "ngram_range": (
            2,
            6,
        ),  # Wide range to capture patterns across different attacks
        "max_features": 20000,  # More features to capture diverse patterns
    }

    return params


def get_model_params(model_type: str) -> Dict[str, Any]:
    """
    Get model parameters based on model type.

    Args:
        model_type: Type of model (logistic, random_forest, svm, mlp, ann)

    Returns:
        params: Model parameters
    """
    if model_type == "logistic":
        return {"max_iter": 1000, "C": 1.0, "class_weight": "balanced"}
    elif model_type == "random_forest":
        return {
            "n_estimators": 200,  # More trees for better generalization
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
            "class_weight": "balanced",
        }
    elif model_type == "svm":
        return {
            "kernel": "linear",
            "C": 1.0,
            "probability": True,
            "class_weight": "balanced",
            "random_state": 42,
        }
    elif model_type == "mlp":
        return {
            "hidden_layer_sizes": (256, 128),  # Larger network for complex patterns
            "max_iter": 1000,
            "random_state": 42,
        }
    elif model_type == "ann":
        return {
            "hidden_sizes": [512, 256, 128],  # Deep network for complex patterns
            "dropout_rate": 0.5,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 15,  # More epochs for better learning
        }
    else:
        return {}


def train_unified_model(
    data_dir: str,
    model_type: str = "ann",  # Default to ANN which handles complexity better
    feature_extractor: str = "tfidf",  # TF-IDF captures important terms better
    output_path: Optional[str] = None,
    device: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> UniversalTrainer:
    """
    Train a unified model for all attack types.

    Args:
        data_dir: Directory containing dataset files for different attacks
        model_type: Type of model (logistic, random_forest, svm, mlp, ann)
        feature_extractor: Type of feature extractor (count, tfidf)
        output_path: Path to save the trained model
        device: Compute device to use (cuda, mps, xpu, cpu_multi, cpu_serial)
        n_jobs: Number of jobs for parallel processing

    Returns:
        trainer: Trained UniversalTrainer instance
    """
    # Start resource tracking
    track_gpu = device == "cuda" and (TORCH_AVAILABLE or GPUTIL_AVAILABLE)
    resource_tracker = ResourceTracker(track_gpu=track_gpu)
    resource_tracker.start()

    try:
        # Convert device string to ComputeDevice enum
        preferred_device = None
        if device:
            if device.lower() == "cuda":
                preferred_device = ComputeDevice.CUDA
            elif device.lower() == "mps":
                preferred_device = ComputeDevice.MPS
            elif device.lower() == "xpu":
                preferred_device = ComputeDevice.XPU
            elif device.lower() == "cpu_multi":
                preferred_device = ComputeDevice.CPU_MULTI
            elif device.lower() == "cpu_serial":
                preferred_device = ComputeDevice.CPU_SERIAL

        # Get feature extractor parameters
        feature_extractor_params = get_unified_feature_extractor_params()

        # Get model parameters
        model_params = get_model_params(model_type)

        # Initialize trainer with custom parameters
        trainer = UniversalTrainer(
            model_type=model_type,
            feature_extractor=feature_extractor,
            model_params=model_params,
            feature_extractor_params=feature_extractor_params,
            preferred_device=preferred_device,
            n_jobs=n_jobs,
        )

        # Load data from all attack types
        X, y, attack_types = load_all_attack_data(data_dir)

        # Train the model
        metrics = trainer.train(X, y, test_size=0.2, random_state=42)

        # Print results
        logger.info(
            f"Unified model training completed with accuracy: {metrics['accuracy']:.4f}"
        )

        # Analyze performance by attack type
        if "y_pred" in metrics and "y_test" in metrics:
            attack_types_test = attack_types.iloc[metrics["y_test"].index]
            for attack_type in attack_types_test.unique():
                mask = attack_types_test == attack_type
                if sum(mask) > 0:
                    from sklearn.metrics import accuracy_score

                    type_accuracy = accuracy_score(
                        metrics["y_test"][mask], metrics["y_pred"][mask]
                    )
                    logger.info(
                        f"Accuracy for {attack_type}: {type_accuracy:.4f} ({sum(mask)} samples)"
                    )

        # Save the model if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            trainer.save(output_path)
            logger.info(f"Unified model saved to: {output_path}")

        return trainer

    finally:
        # Stop resource tracking and print resume
        resource_tracker.stop()
        resource_tracker.print_resume()


def evaluate_unified_model(
    model_path: str, data_dir: str, device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained unified model on test datasets.

    Args:
        model_path: Path to the trained model file
        data_dir: Directory containing test dataset files
        device: Compute device to use (cuda, mps, xpu, cpu_multi, cpu_serial)

    Returns:
        metrics: Evaluation metrics
    """
    # Start resource tracking
    track_gpu = device == "cuda" and (TORCH_AVAILABLE or GPUTIL_AVAILABLE)
    resource_tracker = ResourceTracker(track_gpu=track_gpu)
    resource_tracker.start()

    try:
        # Convert device string to ComputeDevice enum
        preferred_device = None
        if device:
            if device.lower() == "cuda":
                preferred_device = ComputeDevice.CUDA
            elif device.lower() == "mps":
                preferred_device = ComputeDevice.MPS
            elif device.lower() == "xpu":
                preferred_device = ComputeDevice.XPU
            elif device.lower() == "cpu_multi":
                preferred_device = ComputeDevice.CPU_MULTI
            elif device.lower() == "cpu_serial":
                preferred_device = ComputeDevice.CPU_SERIAL

        # Load the model
        trainer = UniversalTrainer.load(model_path, preferred_device=preferred_device)

        # Load test data
        X, y, attack_types = load_all_attack_data(data_dir)

        # Make predictions
        y_pred = trainer.predict(X)

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )

        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)

        # Print overall results
        logger.info(
            "Evaluation results (overall):\n"
            + f"Accuracy: {accuracy:.4f}\n"
            + f"Classification report:\n{classification_report(y, y_pred)}\n"
            + f"Confusion matrix:\n{conf_matrix}"
        )

        # Analyze performance by attack type
        for attack_type in attack_types.unique():
            mask = attack_types == attack_type
            if sum(mask) > 0:
                type_accuracy = accuracy_score(y[mask], y_pred[mask])
                type_report = classification_report(
                    y[mask], y_pred[mask], output_dict=True
                )
                type_conf_matrix = confusion_matrix(y[mask], y_pred[mask])

                logger.info(
                    f"\nResults for {attack_type}:\n"
                    + f"Type report:\n{type_report}\n"
                    + f"Accuracy: {type_accuracy:.4f}\n"
                    + f"Classification report:\n{classification_report(y[mask], y_pred[mask])}\n"
                    + f"Confusion matrix:\n{type_conf_matrix}"
                )

        # Show a few examples
        num_examples = min(10, len(X))
        logger.info("Example predictions:")
        for i in range(num_examples):
            input_text = X.iloc[i]
            attack = attack_types.iloc[i]
            true_label = "Malicious" if y.iloc[i] == 1 else "normal"
            pred_label = "Malicious" if y_pred[i] == 1 else "normal"
            status = "✓" if true_label == pred_label else "✗"

            logger.info(f"{status} Attack type: {attack}")
            logger.info(f"{status} Input: {input_text[:100]}...")
            logger.info(f"   True: {true_label}, Predicted: {pred_label}")

        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": conf_matrix,
            "attack_type_results": {
                attack_type: {
                    "accuracy": accuracy_score(
                        y[attack_types == attack_type],
                        y_pred[attack_types == attack_type],
                    )
                }
                for attack_type in attack_types.unique()
            },
        }

    finally:
        # Stop resource tracking and print resume
        resource_tracker.stop()
        resource_tracker.print_resume()


def predict(
    model_path: str, input_text: str, device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Make predictions on new input text.

    Args:
        model_path: Path to the trained model file
        input_text: Input text to predict
        device: Compute device to use (cuda, mps, xpu, cpu_multi, cpu_serial)

    Returns:
        result: Prediction result
    """
    # Convert device string to ComputeDevice enum
    preferred_device = None
    if device:
        if device.lower() == "cuda":
            preferred_device = ComputeDevice.CUDA
        elif device.lower() == "mps":
            preferred_device = ComputeDevice.MPS
        elif device.lower() == "xpu":
            preferred_device = ComputeDevice.XPU
        elif device.lower() == "cpu_multi":
            preferred_device = ComputeDevice.CPU_MULTI
        elif device.lower() == "cpu_serial":
            preferred_device = ComputeDevice.CPU_SERIAL

    # Load the model
    trainer = UniversalTrainer.load(model_path, preferred_device=preferred_device)

    # Make predictions
    prediction = trainer.predict([input_text])[0]

    # Get probabilities if available
    probabilities = None
    if hasattr(trainer.model, "predict_proba"):
        probabilities = trainer.predict_proba([input_text])[0]

    # Print results
    pred_label = "Malicious" if prediction == 1 else "normal"
    logger.info(f"Prediction: {pred_label}")

    if probabilities is not None:
        if len(probabilities) == 2:
            logger.info(f"Probability of being malicious: {probabilities[1]:.4f}")
        else:
            logger.info(f"Probabilities: {probabilities}")

    return {
        "prediction": prediction,
        "label": pred_label,
        "probabilities": probabilities,
    }


def continue_training_model(
    model_path: str,
    data_dir: str,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> UniversalTrainer:
    """
    Continue training an existing model with new data.

    Args:
        model_path: Path to the trained model file
        data_dir: Directory containing dataset files
        output_path: Path to save the updated model
        device: Compute device to use
        n_jobs: Number of jobs for parallel processing

    Returns:
        trainer: Updated UniversalTrainer instance
    """
    # Start resource tracking
    track_gpu = device == "cuda" and (TORCH_AVAILABLE or GPUTIL_AVAILABLE)
    resource_tracker = ResourceTracker(track_gpu=track_gpu)
    resource_tracker.start()

    try:
        # Convert device string to ComputeDevice enum
        preferred_device = None
        if device:
            if device.lower() == "cuda":
                preferred_device = ComputeDevice.CUDA
            elif device.lower() == "mps":
                preferred_device = ComputeDevice.MPS
            elif device.lower() == "xpu":
                preferred_device = ComputeDevice.XPU
            elif device.lower() == "cpu_multi":
                preferred_device = ComputeDevice.CPU_MULTI
            elif device.lower() == "cpu_serial":
                preferred_device = ComputeDevice.CPU_SERIAL

        # Load the existing model
        trainer = UniversalTrainer.load(model_path, preferred_device=preferred_device)
        logger.info(f"Loaded existing model from {model_path}")

        # Load new training data
        X, y, attack_types = load_all_attack_data(data_dir)

        # Continue training with new data
        logger.info("Continuing training with new data...")
        metrics = trainer.train(X, y, test_size=0.2, random_state=42)

        logger.info(
            f"Updated model training completed with accuracy: {metrics['accuracy']:.4f}"
        )

        # Analyze performance by attack type
        if "y_pred" in metrics and "y_test" in metrics:
            attack_types_test = attack_types.iloc[metrics["y_test"].index]
            for attack_type in attack_types_test.unique():
                mask = attack_types_test == attack_type
                if sum(mask) > 0:
                    from sklearn.metrics import accuracy_score

                    type_accuracy = accuracy_score(
                        metrics["y_test"][mask], metrics["y_pred"][mask]
                    )
                    logger.info(
                        f"Accuracy for {attack_type}: {type_accuracy:.4f} ({sum(mask)} samples)"
                    )

        # Save the updated model if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            trainer.save(output_path)
            logger.info(f"Updated model saved to: {output_path}")

        return trainer

    finally:
        # Stop resource tracking and print resume
        resource_tracker.stop()
        resource_tracker.print_resume()


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description="Universal model training tool for unified web attack detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Command options
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train unified model command
    train_parser = subparsers.add_parser(
        "train", help="Train a unified model for all attack types"
    )
    train_parser.add_argument(
        "--data-dir", required=True, help="Directory containing attack datasets"
    )
    train_parser.add_argument(
        "--model-type",
        default="ann",
        choices=["logistic", "random_forest", "svm", "mlp", "ann"],
        help="Type of model",
    )
    train_parser.add_argument(
        "--feature-extractor",
        default="tfidf",
        choices=["count", "tfidf"],
        help="Type of feature extractor",
    )
    train_parser.add_argument("--output", help="Path to save the trained model")
    train_parser.add_argument(
        "--device",
        choices=["cuda", "mps", "xpu", "cpu_multi", "cpu_serial"],
        help="Compute device to use",
    )
    train_parser.add_argument(
        "--n-jobs", type=int, help="Number of jobs for parallel processing"
    )

    # Continue training command
    continue_parser = subparsers.add_parser(
        "continue-training", help="Continue training an existing model with new data"
    )
    continue_parser.add_argument(
        "--model", required=True, help="Path to existing model file"
    )
    continue_parser.add_argument(
        "--data-dir", required=True, help="Directory containing training datasets"
    )
    continue_parser.add_argument("--output", help="Path to save the updated model")
    continue_parser.add_argument(
        "--device",
        choices=["cuda", "mps", "xpu", "cpu_multi", "cpu_serial"],
        help="Compute device to use",
    )
    continue_parser.add_argument(
        "--n-jobs", type=int, help="Number of jobs for parallel processing"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a unified model")
    eval_parser.add_argument(
        "--model", required=True, help="Path to trained model file"
    )
    eval_parser.add_argument(
        "--data-dir", required=True, help="Directory containing test datasets"
    )
    eval_parser.add_argument(
        "--device",
        choices=["cuda", "mps", "xpu", "cpu_multi", "cpu_serial"],
        help="Compute device to use",
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Make predictions on new data"
    )
    predict_parser.add_argument(
        "--model", required=True, help="Path to trained model file"
    )
    predict_parser.add_argument("--input", required=True, help="Input text to predict")
    predict_parser.add_argument(
        "--device",
        choices=["cuda", "mps", "xpu", "cpu_multi", "cpu_serial"],
        help="Compute device to use",
    )

    # Device info command
    subparsers.add_parser(
        "device-info", help="Display information about available compute devices"
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == "train":
        train_unified_model(
            data_dir=args.data_dir,
            model_type=args.model_type,
            feature_extractor=args.feature_extractor,
            output_path=args.output,
            device=args.device,
            n_jobs=args.n_jobs,
        )
    elif args.command == "continue-training":
        continue_training_model(
            model_path=args.model,
            data_dir=args.data_dir,
            output_path=args.output,
            device=args.device,
            n_jobs=args.n_jobs,
        )
    elif args.command == "evaluate":
        evaluate_unified_model(
            model_path=args.model,
            data_dir=args.data_dir,
            device=args.device,
        )
    elif args.command == "predict":
        predict(
            model_path=args.model,
            input_text=args.input,
            device=args.device,
        )
    elif args.command == "device-info":
        # Print device information
        get_device_manager().print_device_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
