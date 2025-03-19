import re
import joblib


def is_http_request(request):
    """Check if the input appears to be an HTTP request."""
    http_patterns = [
        r"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH) .+ HTTP/\d\.\d",
        r"Host: ",
        r"Content-Length: ",
        r"Content-Type: ",
        r"\r\n\r\n",
    ]

    for pattern in http_patterns:
        if re.search(pattern, request, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def is_regex_payload(payload):
    """Check if the input appears to be a regular expression payload."""
    regex_patterns = [
        r"^\/.+\/[a-z]*$",  # Pattern like /regex/flags
        r"^\(.+\)[+*?]",  # Pattern with repetition quantifiers
        r"\([a-zA-Z0-9]+[+*?]\)[+*?]",  # Nested repetition
    ]

    for pattern in regex_patterns:
        if re.search(pattern, payload):
            return True
    return False


def is_resource_exhaustion_code(code):
    """Check if the code appears to be resource exhaustion code."""
    code_patterns = [
        r"while\s*\(\s*(?:true|1)\s*\)",  # Infinite loop
        r"for\s*\(\s*;{1,2}\s*;?\s*\)",  # Empty for loop (infinite)
        r"\.repeat\s*\(\s*\d{7,}\s*\)",  # Long string repeat
        r"Array\s*\(\s*\d{7,}\s*\)",  # Large array allocation
        r"fork\s*\(\s*\)",  # Process forking
        r"new\s+Array\s*\(\s*\d{7,}\s*\)",  # Large array constructor
    ]

    for pattern in code_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return True
    return False


def is_sql_exhaustion(query):
    """Check if the SQL query is designed to exhaust resources."""
    sql_patterns = [
        r"BENCHMARK\s*\(\s*\d{5,}\s*,",
        # SQL benchmark function with large count
        r"pg_sleep\s*\(\s*\d+\s*\)",  # PostgreSQL sleep function
        r"ORDER\s+BY.+non_indexed",  # Ordering by non-indexed columns
        # Multiple joins (potentially expensive)
        r"JOIN\s+(?:\w+\s+){4,}",
    ]

    for pattern in sql_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


def load_model(model_path):
    """Load a trained model."""
    try:
        model, vectorizer = joblib.load(model_path)
        print("Model successfully loaded!")
        return model, vectorizer
    except FileNotFoundError:
        print("Model file not found!")
        return None, None


def classify_payload(model, vectorizer, payload):
    """Classify a payload using the model."""
    if not model or not vectorizer:
        print("Model not loaded!")
        return None, None

    payload_vectorized = vectorizer.transform([payload])
    prediction = model.predict(payload_vectorized)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(payload_vectorized)[0]
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        return prediction, confidence

    return prediction, None


def explain_payload(payload):
    """Provide explanations for the DoS payload."""
    explanations = []

    patterns = {
        "HTTP Flood": [
            r"(GET|POST|HEAD|OPTIONS) \/[^ ]* HTTP\/\d\.\d.*(GET|POST|HEAD|OPTIONS) \/[^ ]* HTTP\/\d\.\d",
            r"Content-Length: [1-9]\d{6,}",
        ],
        "Slowloris Attack": [
            r"GET .+ HTTP\/\d\.\d\r\nHost: .+\r\nX-[a-zA-Z0-9-]+: [^\n]+\r\n(?!\r\n)",
        ],
        "SlowPOST Attack": [
            r"POST .+ HTTP\/\d\.\d.+Content-Length: \d{4,}\r\n\r\na=",
        ],
        "ReDoS (Regex DoS)": [
            r"\(\w+[+*]\)[+*]",
            r"\(\[.+\][+*]\)[+*]",
            r"\(\w+\|\w+\)[+*]",
        ],
        "XML Billion Laughs": [
            r"<!ENTITY\s+lol\d+\s+\"&lol\d+;.*?&lol\d+;\"",
            r"<!DOCTYPE\s+[^>]+\[\s*<!ENTITY",
        ],
        "HashDoS Attack": [
            r"var\d+=val\d+&var\d+=val\d+&var\d+=val\d+",
        ],
        "ZIP Bomb": [
            r"Content-Type:\s*application\/zip.+\d{8,}",
        ],
        "Fork Bomb": [
            r":\(\)\{\s*:\|:&\s*\};:",
            r"fork_bomb\(\).*fork_bomb\(\)",
            r"while\s*\(\s*\d+\s*\)\s*\{\s*fork\s*\(\s*\)",
        ],
        "Memory Exhaustion": [
            r"Array\(\d{8,}\)",
            r"\.repeat\(\d{8,}\)",
            r"while\(true\).*?push\(new Array",
            r"'\s*'\s*\*\s*10\*\*\d{2,}",
        ],
        "CPU Exhaustion": [
            r"for\s*\(\s*var\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*\d{9,}\s*;",
            r"while\s*\(\s*true\s*\)",
            r"for\s*\(\s*;{1,2}\s*;?\s*\)",
            r"require\(\s*['\"]crypto['\"]\s*\)\.randomBytes\(\s*\d{6,}\s*\)",
        ],
        "File Descriptor Exhaustion": [
            r"for\s*\(.+\d{4,}.+\{\s*socket\s*\(\s*\)",
        ],
    }

    for attack_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, payload, re.IGNORECASE | re.MULTILINE):
                explanations.append(
                    f"Detected pattern matching '{attack_type}' DoS attack"
                )

    if not explanations:
        # Additional checks for common resource-exhausting patterns
        if "SELECT" in payload and is_sql_exhaustion(payload):
            explanations.append(
                "Detected SQL query designed to exhaust database resources"
            )
        elif is_resource_exhaustion_code(payload):
            explanations.append(
                "Detected code snippet designed to exhaust system resources"
            )
        elif len(payload) > 5000:
            explanations.append(
                "Payload is unusually large which may indicate a volumetric DoS attempt"
            )

    if not explanations:
        explanations.append("No known DoS patterns detected")

    return explanations


def main():
    """Main function."""
    model_path = input("Enter path to trained model (.joblib): ")
    model, vectorizer = load_model(model_path)

    while True:
        payload = input("\nEnter payload to analyze (or 'exit' to quit): ").strip()
        if payload.lower() == "exit":
            print("Exiting program.")
            break

        # Determine payload type
        payload_type = "Unknown"
        if is_http_request(payload):
            payload_type = "HTTP Request"
        elif is_regex_payload(payload):
            payload_type = "Regular Expression"
        elif is_resource_exhaustion_code(payload):
            payload_type = "Code Snippet"
        elif payload.startswith("<?xml") or "<!DOCTYPE" in payload:
            payload_type = "XML Data"
        elif payload.startswith("SELECT") or "FROM" in payload:
            payload_type = "SQL Query"

        print(f"\nPayload type detected: {payload_type}")

        if model and vectorizer:
            prediction, confidence = classify_payload(model, vectorizer, payload)
            if prediction == 1:
                if confidence:
                    print(
                        f"MALICIOUS - Potential DoS payload detected (Confidence: {confidence:.2f})"
                    )
                else:
                    print("MALICIOUS - Potential DoS payload detected")
            else:
                if confidence:
                    print(f"normal - Normal payload (Confidence: {confidence:.2f})")
                else:
                    print("normal - Normal payload")

        explanations = explain_payload(payload)
        print("\nAnalysis:")
        for explanation in explanations:
            print(f"- {explanation}")

        # Resource impact assessment
        print("\nResource Impact Assessment:")
        if len(payload) > 10000:
            print("- HIGH BANDWIDTH impact - Payload size exceeds 10KB")
        elif len(payload) > 1000:
            print("- MODERATE BANDWIDTH impact - Payload size exceeds 1KB")

        if any(
            pattern in payload
            for pattern in ["while(true)", "for(;;)", "BENCHMARK", "pg_sleep"]
        ):
            print("- HIGH CPU impact - Contains infinite loops or blocking operations")

        if any(
            pattern in payload
            for pattern in ["Array(", "new Array", "repeat(", " * 10**"]
        ):
            print("- HIGH MEMORY impact - Contains memory exhaustion patterns")

        if "fork(" in payload or ":(){ :|:& };" in payload:
            print("- HIGH PROCESS impact - Contains process spawning patterns")


if __name__ == "__main__":
    main()
