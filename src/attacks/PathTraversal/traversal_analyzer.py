import re
import joblib
import urllib.parse


def is_traversal_pattern(path):
    """Check if the path contains directory traversal patterns."""
    traversal_patterns = [
        r"\.\.\/",  # ../
        r"\.\.\\",  # ..\
        r"%2e%2e\/",  # %2e%2e/
        r"%2e%2e\\",  # %2e%2e\
        r"\.\.%2f",  # ..%2f
        r"\.\.%5c",  # ..%5c
        r"%252e%252e",  # double-encoded ..
        r"\.\.%c0%af",  # UTF-8 encoded
        r"\.\.%ef%bc%8f",  # Wide character encoded
        r"\.{2,}[\/\\]",  # Multiple dots followed by slash
        r"%c0%ae",  # Unicode encoded dot
    ]

    for pattern in traversal_patterns:
        if re.search(pattern, path, re.IGNORECASE):
            return True

    # Check for consecutive back references (../.., ..\..\, etc.)
    if re.search(r"(\.\.[\\/]){2,}", path, re.IGNORECASE):
        return True

    # Check for many consecutive dots (which could be used to bypass filters)
    if re.search(r"\.{3,}", path):
        return True

    # Check if path contains %00 (null byte)
    if "%00" in path:
        return True

    return False


def is_path_targeting_sensitive_file(path):
    """Check if the path is trying to access sensitive files."""
    sensitive_files = [
        r"etc[\\\/]passwd",
        r"etc[\\\/]shadow",
        r"etc[\\\/]hosts",
        r"etc[\\\/]ssh",
        r"windows[\\\/]win\.ini",
        r"boot\.ini",
        r"system32[\\\/]drivers",
        r"windows[\\\/]repair[\\\/]sam",
        r"web\.config",
        r"wp-config\.php",
        r"config\.php",
        r"config\.inc\.php",
        r"\.env",
        r"\.git",
        r"\.htaccess",
        r"\.ssh[\\\/]id_rsa",
        r"\.ssh[\\\/]authorized_keys",
        r"proc[\\\/]self",
        r"var[\\\/]log",
        r"var[\\\/]www",
        r"httpd\.conf",
        r"nginx\.conf",
    ]

    # URL decode the path to check for encoded sensitive files
    decoded_path = urllib.parse.unquote(path)

    for pattern in sensitive_files:
        if re.search(pattern, path, re.IGNORECASE) or re.search(
            pattern, decoded_path, re.IGNORECASE
        ):
            return True, pattern.replace(r"[\\\/]", "/")

    return False, ""


def has_path_manipulation(path):
    """Check if the path contains manipulation techniques."""
    manipulation_patterns = [
        r"\.{5,}[\/\\]",  # Many dots
        r"[\/\\]{2,}",  # Multiple slashes
        r"[\/\\]\.+[\/\\]",  # Slash followed by dots followed by slash
        r"%00",  # Null byte
        r"%25(?:[0-9a-fA-F]{2}){2,}",  # Double encoding
    ]

    for pattern in manipulation_patterns:
        if re.search(pattern, path, re.IGNORECASE):
            return True

    return False


def load_model(model_path):
    """Load a trained model and vectorizer from a file."""
    try:
        model, vectorizer = joblib.load(model_path)
        print("Model successfully loaded!")
        return model, vectorizer
    except FileNotFoundError:
        print("Model file not found!")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def classify_path(model, vectorizer, path):
    """Classify a path using the trained model."""
    if not model or not vectorizer:
        print("Model not loaded!")
        return None, None  # Return a tuple with None values

    path_vectorized = vectorizer.transform([path])
    prediction = model.predict(path_vectorized)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(path_vectorized)[0]
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        return prediction, confidence

    return prediction, None


def explain_path(path):
    """Provide detailed explanations about a potential path traversal attack."""
    explanations = []

    # URL decode for analysis
    decoded_path = urllib.parse.unquote(path)

    # If the path was URL encoded, note it
    if decoded_path != path:
        explanations.append(f"URL encoding detected: {path} decodes to {decoded_path}")

    # Check for double encoding
    if re.search(r"%25(?:[0-9a-fA-F]{2})+", path):
        double_decoded = urllib.parse.unquote(decoded_path)
        if double_decoded != decoded_path:
            explanations.append(
                f"Double URL encoding detected: {decoded_path} further decodes to {double_decoded}"
            )
            decoded_path = double_decoded

    # Check for directory traversal sequences
    if re.search(r"\.\.[\\/]", decoded_path):
        count = len(re.findall(r"\.\.[\\/]", decoded_path))
        explanations.append(
            f"Directory traversal detected: {count} back-reference(s) found"
        )

    # Check if trying to access a sensitive file
    is_sensitive, file_pattern = is_path_targeting_sensitive_file(decoded_path)
    if is_sensitive:
        explanations.append(f"Attempt to access sensitive system file: {file_pattern}")

    # Check for null byte injection
    if "%00" in path:
        file_ext = path.split("%00")[-1] if len(path.split("%00")) > 1 else ""
        explanations.append(
            f"Null byte injection detected, possibly trying to bypass file extension filtering: {file_ext}"
        )

    # Check for other evasion techniques
    if re.search(r"\.{3,}", decoded_path):
        explanations.append("Multiple dots detected, possible filter evasion technique")

    if re.search(r"[\/\\]{2,}", decoded_path):
        explanations.append(
            "Multiple consecutive slashes detected, possible filter evasion technique"
        )

    # Analysis for Unicode/UTF-8 encoding evasion
    if any(
        pattern in path.lower()
        for pattern in ["%c0%af", "%ef%bc%8f", "%c0%ae", "%c0%2f", "%e0%80%2f"]
    ):
        explanations.append("Unicode/UTF-8 encoding evasion technique detected")

    if not explanations:
        explanations.append("No specific path traversal patterns detected")

    return explanations


def normalize_path(path):
    """Normalize a path by removing traversal sequences."""
    # URL decode first
    decoded_path = urllib.parse.unquote(path)

    # Check for double encoding
    if "%25" in decoded_path:
        decoded_path = urllib.parse.unquote(decoded_path)

    # Split path by directory separators
    parts = re.split(r"[\/\\]", decoded_path)

    # Process the path components
    normalized_parts = []
    for part in parts:
        if part == "" or part == ".":
            continue  # Skip empty parts and current directory references
        elif part == "..":
            if normalized_parts:  # If we have parts to go back from
                normalized_parts.pop()  # Remove the last part
        else:
            normalized_parts.append(part)

    # Reconstruct the path
    normalized_path = "/" + "/".join(normalized_parts)
    return normalized_path


def main():
    """Main function to run the directory traversal analyzer."""
    model_path = input("Enter path to trained model (.joblib) or press Enter to skip: ")
    model, vectorizer = None, None

    if model_path:
        model, vectorizer = load_model(model_path)

    while True:
        path = input("\nEnter path to analyze (or 'exit' to quit): ").strip()
        if path.lower() == "exit":
            print("Exiting program.")
            break

        print("\nPath analysis:")

        # Perform basic pattern detection
        is_traversal = is_traversal_pattern(path)
        if is_traversal:
            print("⚠️  PATH TRAVERSAL PATTERN DETECTED!")
        else:
            print("No basic traversal pattern detected.")

        # Check for sensitive file targeting
        is_sensitive, file_pattern = is_path_targeting_sensitive_file(path)
        if is_sensitive:
            print(f"⚠️  SENSITIVE FILE TARGET DETECTED: {file_pattern}")

        # Check for path manipulation techniques
        if has_path_manipulation(path):
            print("⚠️  PATH MANIPULATION TECHNIQUES DETECTED!")

        # Model-based classification if available
        if model and vectorizer:
            prediction, confidence = classify_path(model, vectorizer, path)
            if prediction == 1:
                if confidence:
                    print(
                        f"🚨 MALICIOUS - Path classified as a traversal attack (Confidence: {confidence:.2f})"
                    )
                else:
                    print("🚨 MALICIOUS - Path classified as a traversal attack")
            else:
                if confidence is not None:  # Check that confidence is not None
                    print(
                        f"✅ BENIGN - Path classified as normal (Confidence: {confidence:.2f})"
                    )
                else:
                    print("✅ BENIGN - Path classified as normal")

        # Detailed explanations
        print("\nDetailed Analysis:")
        explanations = explain_path(path)
        for explanation in explanations:
            print(f"- {explanation}")

        # Show normalized path
        normalized = normalize_path(path)
        print(f"\nNormalized path: {normalized}")

        # Risk assessment
        risk_level = "LOW"
        if is_traversal and is_sensitive:
            risk_level = "HIGH"
        elif is_traversal or is_sensitive:
            risk_level = "MEDIUM"

        print(f"\nOverall risk assessment: {risk_level}")


if __name__ == "__main__":
    main()
