import requests

BASE_URL = "http://localhost:8000"


def get_token():
    """Get authentication token from API."""
    response = requests.post(
        f"{BASE_URL}/token", data={"username": "admin", "password": "secret"}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print(f"Failed to get token: {response.text}")
        return None


def submit_event(token, event_data):
    """Submit a security event to the API."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(f"{BASE_URL}/events", headers=headers, json=event_data)
    return response


def main():
    # Get authentication token
    token = get_token()
    if not token:
        print("Authentication failed. Exiting.")
        return

    print(f"Successfully authenticated, token: {token[:10]}...")

    # Create a test event
    event = {
        "source_ip": "192.168.1.100",
        "destination_ip": "192.168.1.5",
        "event_type": "connection",
        "source_type": "Firewall",
        "payload": {"protocol": "TCP", "port": 443, "flags": "SYN", "packet_count": 5},
    }

    # Submit the event
    response = submit_event(token, event)
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.json()}")


if __name__ == "__main__":
    main()
