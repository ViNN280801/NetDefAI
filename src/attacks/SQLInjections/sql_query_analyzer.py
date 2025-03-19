import re
import joblib


def is_sql_query(query):
    sql_keywords = [
        r"SELECT",
        r"INSERT INTO",
        r"UPDATE",
        r"DELETE FROM",
        r"DROP TABLE",
        r"CREATE TABLE",
        r"ALTER TABLE",
        r"TRUNCATE TABLE",
        r"WHERE",
        r"FROM",
        r"JOIN",
        r"GROUP BY",
        r"ORDER BY",
        r"LIMIT",
        r"UNION",
        r"VALUES",
        r"SET",
    ]

    for keyword in sql_keywords:
        if re.search(rf"\b{keyword}\b", query, re.IGNORECASE):
            return True
    return False


def is_query_allowed(query):
    forbidden_keywords = [r"DROP TABLE", r"DELETE FROM", r"TRUNCATE TABLE"]

    for keyword in forbidden_keywords:
        if re.search(rf"\b{keyword}\b", query, re.IGNORECASE):
            return False, keyword
    return True, ""


def load_model(model_path):
    try:
        model, vectorizer = joblib.load(model_path)
        print("Модель успешно загружена!")
        return model, vectorizer
    except FileNotFoundError:
        print("Файл модели не найден!")
        return None, None


def classify_query(model, vectorizer, query):
    if not model or not vectorizer:
        print("Модель не загружена!")
        return

    query_vectorized = vectorizer.transform([query])
    prediction = model.predict(query_vectorized)[0]
    return prediction


def explain_query(query):
    explanations = []

    patterns = {
        "Generic": [
            r" OR ",
            r" AND ",
            r"--",
            r";",
            r"\/\*",
        ],
        "Union-based": [
            r" UNION ",
            r"UNION SELECT",
        ],
        "Boolean-based": [
            r" OR 1=1",
            r" AND 1=2",
            r"OR TRUE",
        ],
        "Error-based": [
            r"' OR '",
            r"\"",
            r"CHAR\(",
        ],
        "Stacked Queries": [
            r";",
            r"; DROP",
        ],
        "Time-based Blind": [
            r"SLEEP\(",
            r"BENCHMARK\(",
        ],
        "Out-of-Band": [
            r"LOAD_FILE\(",
            r"OUTFILE",
        ],
        "Destructive Queries": [
            r"DROP TABLE",
            r"DROP COLUMN",
            r"DELETE FROM",
            r"TRUNCATE TABLE",
        ],
    }

    for attack_type, patterns_list in patterns.items():
        for pattern in patterns_list:
            if re.search(pattern, query, re.IGNORECASE):
                if attack_type == "Destructive Queries":
                    explanations.append(
                        f"Detected '{pattern}' — Dangerous query: {attack_type}. This may lead to data deletion."
                    )
                else:
                    explanations.append(
                        f"Detected '{pattern}' — SQL injection type: {attack_type}"
                    )

    if not explanations:
        explanations.append("Query does not contain known SQL injection patterns.")
    return explanations


def main():
    model_path = input("Enter the path to the trained model (.joblib): ")
    model, vectorizer = load_model(model_path)

    if not model or not vectorizer:
        print("An error occurred while loading the model, exiting the program...")
        return

    while True:
        query = input("\nEnter a SQL query (or 'exit' to exit): ").strip()
        if query.lower() == "exit":
            print("Exiting the program.")
            break

        if not is_sql_query(query):
            print("This is not a SQL query.")
            continue

        is_allowed, forbidden_keyword = is_query_allowed(query)
        if not is_allowed:
            print(
                f"This is a syntactically correct SQL query, but it contains forbidden keywords: {forbidden_keyword}"
            )
            continue

        prediction = classify_query(model, vectorizer, query)
        if prediction == 1:
            print("DANGEROUS query!")
        else:
            print("SAFE query.")

        explanations = explain_query(query)
        print("Reasons:")
        for explanation in explanations:
            print(f"- {explanation}")


if __name__ == "__main__":
    main()
