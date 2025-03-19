import re
import joblib


def is_html_content(content):
    """Check if the content contains HTML or JS elements."""
    html_patterns = [
        r"<(?:script|img|svg|iframe|a|body|div|marquee|details|video|audio|object|embed|form|button|input|meta|base)",
        r"javascript:",
        r"onerror=",
        r"onload=",
        r"alert\(",
        r"document\.cookie",
        r"eval\(",
    ]

    for pattern in html_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False


def is_content_allowed(content):
    """Check if the content contains potentially dangerous elements."""
    dangerous_patterns = [
        r"<script",
        r"javascript:",
        r"eval\(",
        r"document\.cookie",
        r"document\.location",
        r"window\.location",
        r"fetch\(",
        r"\.src\s*=",
        r"atob\(",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return False, pattern
    return True, ""


def load_model(model_path):
    """Load the trained model and vectorizer."""
    try:
        model, vectorizer = joblib.load(model_path)
        print("Model successfully loaded!")
        return model, vectorizer
    except FileNotFoundError:
        print("Model file not found!")
        return None, None


def classify_content(model, vectorizer, content):
    """Classify content as malicious or normal using the model."""
    if not model or not vectorizer:
        print("Model not loaded!")
        return

    content_vectorized = vectorizer.transform([content])
    prediction = model.predict(content_vectorized)[0]
    return prediction


def explain_content(content):
    """Provide explanations for suspicious content."""
    explanations = []

    patterns = {
        "Basic XSS": [
            r"<script>.*?</script>",
            r"<img[^>]+onerror=",
            r"<svg[^>]+onload=",
            r"alert\(",
            r"javascript:",
        ],
        "Cookie Theft": [
            r"document\.cookie",
            r"fetch\([^)]*cookie",
            r"\.src\s*=\s*['\"]\s*http.*?\+\s*document\.cookie",
            r"window\.location\s*=\s*['\"]\s*http.*?\+\s*document\.cookie",
        ],
        "DOM Manipulation": [
            r"document\.write",
            r"document\.createElement",
            r"\.innerHTML",
            r"\.outerHTML",
        ],
        "Event Handler Injection": [
            r"on\w+\s*=",  # onload, onclick, etc.
            r"<\w+[^>]+on\w+\s*=",
        ],
        "Encoded Payloads": [
            r"eval\(atob\(",
            r"base64",
            r"data:text/html;base64",
            r"\\u00",
        ],
        "Remote Script Inclusion": [
            r"<script\s+src=",
            r"\bsrc\s*=\s*['\"](?:https?:)?//",
        ],
    }

    for attack_type, patterns_list in patterns.items():
        for pattern in patterns_list:
            if re.search(pattern, content, re.IGNORECASE):
                explanations.append(
                    f"Detected '{pattern}' — XSS Attack Type: {attack_type}"
                )

    if not explanations:
        explanations.append("Content does not contain known XSS patterns.")
    return explanations


def main():
    """Main function to run the XSS analyzer."""
    model_path = input("Enter the path to the trained model (.joblib): ")
    model, vectorizer = load_model(model_path)

    if not model or not vectorizer:
        print("Error loading model, exiting program...")
        return

    while True:
        content = input(
            "\nEnter HTML/JS content to analyze (or 'exit' to quit): "
        ).strip()
        if content.lower() == "exit":
            print("Exiting program.")
            break

        if not is_html_content(content):
            print("This doesn't appear to be HTML/JS content.")
            continue

        is_allowed, forbidden_pattern = is_content_allowed(content)
        if not is_allowed:
            print(
                f"This content contains potentially dangerous elements: {forbidden_pattern}"
            )

        prediction = classify_content(model, vectorizer, content)
        if prediction == 1:
            print("MALICIOUS content detected!")
        else:
            print("normal content.")

        explanations = explain_content(content)
        print("Details:")
        for explanation in explanations:
            print(f"- {explanation}")


if __name__ == "__main__":
    main()
