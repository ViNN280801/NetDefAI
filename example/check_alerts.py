import time
import requests

BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "secret"


def get_token():
    """Get authentication token."""
    try:
        response = requests.post(
            f"{BASE_URL}/token",
            data={"username": USERNAME, "password": PASSWORD},
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting token: {e}")
    return None


def check_health():
    """Check system health status."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error checking health: {e}")
    return None


def check_metrics():
    """Get Prometheus metrics."""
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Metrics check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error checking metrics: {e}")
    return None


def analyze_metrics(metrics_text):
    """Parse metrics to find evidence of DDoS detection."""
    if not metrics_text:
        return {}

    results = {}

    # Extract key metrics
    for line in metrics_text.split("\n"):
        if line.startswith("#"):
            continue

        if "api_request_count" in line:
            parts = line.split()
            if len(parts) >= 2:
                label = parts[0]
                value = parts[1]
                if 'status_code="429"' in label:  # Too Many Requests
                    results["rate_limited_requests"] = float(value)
                elif 'endpoint="/events"' in label and 'status_code="202"' in label:
                    results["successful_events"] = float(value)

        if "events_processed_total" in line:
            parts = line.split()
            if len(parts) >= 2:
                label = parts[0]
                value = parts[1]
                if 'status="success"' in label:
                    results["processed_events"] = float(value)
                elif 'status="error"' in label:
                    results["error_events"] = float(value)

    return results


def main():
    print("=== Checking System Alerts ===")

    # Check system health
    print("\nChecking system health...")
    health = check_health()
    if health:
        print(f"System health: {health['status']}")
        for service, status in health["services"].items():
            print(f"- {service}: {status}")

    # Wait a moment for metrics to update
    print("\nWaiting for metrics to update...")
    time.sleep(2)

    # Check metrics
    print("\nChecking metrics...")
    metrics = check_metrics()
    if metrics:
        print("Metrics retrieved successfully")

        # Analyze metrics
        analysis = analyze_metrics(metrics)
        print("\n=== DDoS Detection Analysis ===")

        if (
            "rate_limited_requests" in analysis
            and analysis["rate_limited_requests"] > 0
        ):
            print(
                f"✅ DETECTED: System rate-limited {analysis['rate_limited_requests']} requests"
            )
        else:
            print("❌ NOT DETECTED: No rate-limiting observed")

        if "successful_events" in analysis:
            print(f"Successful event submissions: {analysis['successful_events']}")

        if "processed_events" in analysis:
            print(f"Successfully processed events: {analysis['processed_events']}")

        if "error_events" in analysis:
            print(f"Events with processing errors: {analysis['error_events']}")

        # Check if anything suggests DDoS protection worked
        if any(k in analysis and analysis[k] > 0 for k in ["rate_limited_requests"]):
            print("\n✅ DDoS protection appears to be active")
        else:
            print("\n❓ No clear evidence of DDoS protection active")
            print("This could mean:")
            print("- DDoS protection is working at a different layer")
            print("- The attack wasn't intense enough to trigger protection")
            print("- DDoS protection might not be fully implemented")

    print("\nFor more detailed monitoring, check:")
    print("- Prometheus dashboard: http://localhost:9090")
    print("- Grafana dashboard: http://localhost:3000 (admin/admin)")
    print("- RabbitMQ dashboard: http://localhost:15672 (guest/guest)")


if __name__ == "__main__":
    main()
