#!/usr/bin/env python3
"""
Model Architecture Visualization Tool

This script loads a trained model from a .joblib file and displays its structure
in a human-readable format. It supports various model types including sklearn models,
neural networks, and custom UniversalTrainer models.
"""

import os
import sys
import joblib
import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def convert_to_serializable(obj: Any) -> Any:
    """Convert a complex object to a JSON-serializable format."""
    if hasattr(obj, "__dict__"):
        return {
            "type": type(obj).__name__,
            "module": type(obj).__module__,
            "attributes": {
                k: convert_to_serializable(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_") and not callable(v)
            },
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, np.ndarray):
        return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
    else:
        return str(obj)


def visualize_model_structure(model_data: Dict[str, Any], model_path: str) -> None:
    """
    Visualize and print the structure of the model in a readable format.

    Args:
        model_data: Model data dictionary
        model_path: Path to the model file
    """
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)

    # Identify model type
    model_type = None
    feature_extractor_type = None

    # Check if it's a UniversalTrainer model
    if isinstance(model_data, dict) and "model_type" in model_data:
        model_type = model_data.get("model_type")
        feature_extractor_type = model_data.get("feature_extractor_type")

        print(f"\nModel Type: {model_type}")
        if feature_extractor_type:
            print(f"Feature Extractor: {feature_extractor_type}")

        # Check if it's trained
        if "trained" in model_data:
            print(f"Model trained: {'Yes' if model_data['trained'] else 'No'}")

        # Extract ANN details for PyTorch models
        if model_type == "ann" and "ann_params" in model_data:
            print("\nNeural Network Architecture:")
            print(
                f"  Hidden layers: {model_data['ann_params'].get('hidden_sizes', [])}"
            )
            print(f"  Dropout rate: {model_data['ann_params'].get('dropout_rate', 0)}")
            print(
                f"  Learning rate: {model_data['ann_params'].get('learning_rate', 0)}"
            )
            print(f"  Batch size: {model_data['ann_params'].get('batch_size', 0)}")
            print(f"  Epochs: {model_data['ann_params'].get('epochs', 0)}")

            # Try to load the PyTorch model part if it exists
            torch_path = model_data.get("_torch_path", "")
            if not torch_path and isinstance(model_data, dict):
                base_path = os.path.dirname(model_path)
                file_name = os.path.basename(model_path).replace(
                    ".joblib", "_pytorch_model.pt"
                )
                torch_path = os.path.join(base_path, file_name)

            if TORCH_AVAILABLE and os.path.exists(torch_path):
                try:
                    # Load and display PyTorch model
                    torch_model = torch.load(
                        torch_path, map_location=torch.device("cpu")
                    )
                    print("\nPyTorch Model Parameters:")
                    for name, param in torch_model.items():
                        if isinstance(param, torch.Tensor):
                            print(
                                f"  {name}: Tensor(shape={list(param.shape)}, dtype={param.dtype})"
                            )
                except Exception as e:
                    print(f"\nError loading PyTorch model: {e}")

        # Handle sklearn models
        elif (
            SKLEARN_AVAILABLE
            and "model" in model_data
            and hasattr(model_data["model"], "get_params")
        ):
            print("\nModel Parameters:")
            try:
                params = model_data["model"].get_params()
                for key, value in params.items():
                    if not callable(value) and not key.startswith("_"):
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"  Error extracting parameters: {e}")

        # Feature extractor details
        if "feature_extractor" in model_data and SKLEARN_AVAILABLE:
            if isinstance(
                model_data["feature_extractor"], (CountVectorizer, TfidfVectorizer)
            ):
                fe = model_data["feature_extractor"]
                print("\nFeature Extractor Details:")
                print(f"  Type: {type(fe).__name__}")
                print(
                    f"  Vocabulary size: {len(fe.vocabulary_) if hasattr(fe, 'vocabulary_') else 'Not trained'}"
                )

                if hasattr(fe, "get_params"):
                    try:
                        fe_params = fe.get_params()
                        print("  Parameters:")
                        for key, value in fe_params.items():
                            if not callable(value) and not key.startswith("_"):
                                print(f"    {key}: {value}")
                    except Exception as e:
                        print(f"  Error extracting parameters: {e}")

    # If basic structure detection failed, use generic approach
    if model_type is None:
        # Convert to serializable format for cleaner printing
        serializable_data = convert_to_serializable(model_data)
        print("\nGeneric Model Structure:")
        pprint.pprint(serializable_data, indent=2, width=100, depth=3)

    print("\n" + "=" * 60)


def plot_model_graphs(
    model_data: Dict[str, Any], output_dir: Optional[str] = None
) -> None:
    """
    Generate visualizations for the model if possible.

    Args:
        model_data: Model data dictionary
        output_dir: Directory to save visualizations
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if it's a classifier with feature importances
    if SKLEARN_AVAILABLE and "model" in model_data:
        model = model_data.get("model")

        # Plot feature importances if available
        if model is not None and hasattr(model, "feature_importances_"):
            try:
                feature_importances = model.feature_importances_
                if len(feature_importances) > 0:
                    # Get feature names if possible
                    feature_names = None
                    if "feature_extractor" in model_data and hasattr(
                        model_data["feature_extractor"], "get_feature_names_out"
                    ):
                        try:
                            feature_names = model_data[
                                "feature_extractor"
                            ].get_feature_names_out()
                        except Exception as e:
                            print(f"Error getting feature names: {e}")
                            return

                    # Limit to top 20 features for readability
                    n_features = min(20, len(feature_importances))
                    indices = np.argsort(feature_importances)[-n_features:]

                    plt.figure(figsize=(10, 6))
                    plt.title("Feature Importances")
                    plt.barh(range(n_features), feature_importances[indices])

                    if feature_names is not None and len(feature_names) >= n_features:
                        plt.yticks(
                            range(n_features), [str(feature_names[i]) for i in indices]
                        )
                    else:
                        plt.yticks(range(n_features), [str(i) for i in indices])

                    plt.tight_layout()

                    if output_dir:
                        plt.savefig(os.path.join(output_dir, "feature_importances.png"))
                        print(
                            f"Feature importance plot saved to {os.path.join(output_dir, 'feature_importances.png')}"
                        )
                    else:
                        plt.show()
            except Exception as e:
                print(f"Error creating feature importance plot: {e}")

    # For neural networks, visualize the architecture if possible
    if model_data.get("model_type") == "ann" and "ann_params" in model_data:
        try:
            hidden_sizes = model_data["ann_params"].get("hidden_sizes", [])

            if hidden_sizes:
                # Create a basic architecture visualization
                input_size = model_data.get("input_size", 0)
                all_sizes = (
                    [input_size] + hidden_sizes + [1]
                )  # [input] + [hidden layers] + [output]

                if all_sizes[0] > 0:  # Make sure we have valid sizes
                    plt.figure(figsize=(10, 6))

                    # Create the architecture plot
                    for i, size in enumerate(all_sizes):
                        label = (
                            "Input"
                            if i == 0
                            else "Output" if i == len(all_sizes) - 1 else f"Hidden {i}"
                        )
                        size_display = min(size, 100)  # Cap for display purposes

                        # Draw the layer
                        y_positions = np.linspace(
                            0, size_display, min(size_display, 10)
                        )
                        for y in y_positions:
                            plt.scatter(i, y, s=100, color="skyblue", zorder=2)

                        # Draw more neurons indicator if needed
                        if size > 10:
                            plt.text(i, -0.5, f"({size} neurons)", ha="center")

                        # Layer labels
                        plt.text(i, size_display + 0.5, label, ha="center")

                    # Connect the layers
                    for i in range(len(all_sizes) - 1):
                        size1 = min(all_sizes[i], 100)
                        size2 = min(all_sizes[i + 1], 100)
                        y1_positions = np.linspace(0, size1, min(size1, 10))
                        y2_positions = np.linspace(0, size2, min(size2, 10))

                        # Draw a subset of connections to avoid cluttering
                        for idx1, y1 in enumerate(y1_positions):
                            for idx2, y2 in enumerate(y2_positions):
                                if (idx1 + idx2) % 2 == 0:  # Only draw some connections
                                    plt.plot(
                                        [i, i + 1],
                                        [y1, y2],
                                        "gray",
                                        alpha=0.3,
                                        linewidth=0.5,
                                    )

                    plt.grid(False)
                    plt.axis("off")
                    plt.title("Neural Network Architecture")

                    if output_dir:
                        plt.savefig(os.path.join(output_dir, "nn_architecture.png"))
                        print(
                            "Neural network architecture plot saved to "
                            f"{os.path.join(output_dir, 'nn_architecture.png')}"
                        )
                    else:
                        plt.show()
        except Exception as e:
            print(f"Error creating neural network architecture visualization: {e}")


def analyze_model(model_path: str, output_dir: Optional[str] = None) -> None:
    """
    Load and analyze the model architecture from a .joblib file.

    Args:
        model_path: Path to the .joblib model file
        output_dir: Directory to save visualizations
    """
    try:
        print(f"Loading model from {model_path}...")
        model_data = joblib.load(model_path)

        print("Model loaded successfully.")

        # Add the model path in case we need to find related files
        if isinstance(model_data, dict):
            model_data["_model_path"] = model_path

        # Display model information
        visualize_model_structure(model_data, model_path)

        # Create visualizations
        if output_dir:
            plot_model_graphs(model_data, output_dir)
        else:
            # Ask user if they want to display plots
            display_plots = (
                input("\nDo you want to display model visualizations? (y/n): ")
                .lower()
                .strip()
            )
            if display_plots == "y":
                plot_model_graphs(model_data)

    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing model: {e}")
        sys.exit(1)


def main():
    """Parse arguments and execute model analysis."""
    parser = argparse.ArgumentParser(
        description="Model Architecture Visualization Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model", required=True, help="Path to the .joblib model file to analyze"
    )
    parser.add_argument("--output-dir", help="Directory to save visualization images")

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)

    # Check if it's a .joblib file
    if not args.model.endswith(".joblib"):
        print(
            "Warning: The model file doesn't have a .joblib extension. It may not load correctly."
        )

    analyze_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()
