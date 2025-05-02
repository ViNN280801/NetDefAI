import time
import random
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor

# Target settings
BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "secret"

# Load SQLi patterns from the file
with open("patterns/sql_injections.txt", "r") as f:
    SQLI_PATTERNS = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

# Attack parameters
DEFAULT_THREADS = 5
DEFAULT_REQUESTS_PER_THREAD = 20
DEFAULT_SLEEP_MIN = 0.2
DEFAULT_SLEEP_MAX = 1.0


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


def generate_sqli_event():
    """Generate a security event with SQLi payload."""
    source_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    sql_payload = random.choice(SQLI_PATTERNS)

    return {
        "source_ip": source_ip,
        "event_type": "auth",  # Most common target for SQLi
        "source_type": "SQLi Tester",
        "payload": {
            "username": sql_payload,
            "password": "test",
            "query": f"SELECT * FROM users WHERE username = '{sql_payload}'",
        },
    }


def attack_worker(token, requests_count, sleep_min, sleep_max, worker_id):
    """Worker function that sends SQLi requests."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    success = 0
    failed = 0
    blocked = 0

    print(f"Worker {worker_id} started - sending {requests_count} SQLi requests")

    for i in range(requests_count):
        try:
            event = generate_sqli_event()
            response = requests.post(
                f"{BASE_URL}/events", headers=headers, json=event, timeout=5
            )

            if response.status_code == 202:
                success += 1
            elif response.status_code == 429:  # Blocked
                blocked += 1
            else:
                failed += 1

            sleep_time = random.uniform(sleep_min, sleep_max)
            time.sleep(sleep_time)

        except requests.exceptions.RequestException as e:
            failed += 1
            print(f"Error sending request: {e}")

    print(
        f"Worker {worker_id} finished - Success: {success}, Blocked: {blocked}, Failed: {failed}"
    )
    return success, blocked, failed


def main():
    parser = argparse.ArgumentParser(description="SQLi Test Tool for API Gateway")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of threads (default: {DEFAULT_THREADS})",
    )
    parser.add_argument(
        "-r",
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS_PER_THREAD,
        help=f"Requests per thread (default: {DEFAULT_REQUESTS_PER_THREAD})",
    )
    parser.add_argument(
        "-s",
        "--sleep-min",
        type=float,
        default=DEFAULT_SLEEP_MIN,
        help=f"Minimum sleep between requests (default: {DEFAULT_SLEEP_MIN})",
    )
    parser.add_argument(
        "-m",
        "--sleep-max",
        type=float,
        default=DEFAULT_SLEEP_MAX,
        help=f"Maximum sleep between requests (default: {DEFAULT_SLEEP_MAX})",
    )
    args = parser.parse_args()

    print("=== SQLi Test Tool for API Gateway ===")
    print(f"Target: {BASE_URL}")
    print(f"Threads: {args.threads}")
    print(f"Requests per thread: {args.requests}")
    print(f"Total SQLi patterns loaded: {len(SQLI_PATTERNS)}")

    print("\nGetting authentication token...")
    token = get_token()
    if not token:
        print("Failed to get token. Exiting.")
        return

    print("Starting SQLi test...")
    start_time = time.time()

    total_success = 0
    total_blocked = 0
    total_failed = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for i in range(args.threads):
            future = executor.submit(
                attack_worker,
                token,
                args.requests,
                args.sleep_min,
                args.sleep_max,
                i + 1,
            )
            futures.append(future)

        for future in futures:
            success, blocked, failed = future.result()
            total_success += success
            total_blocked += blocked
            total_failed += failed

    elapsed_time = time.time() - start_time
    requests_per_second = (total_success + total_blocked + total_failed) / elapsed_time

    print("\n=== SQLi Test Results ===")
    print(f"Test duration: {elapsed_time:.2f} seconds")
    print(f"Successful requests: {total_success}")
    print(f"Blocked requests: {total_blocked}")
    print(f"Failed requests: {total_failed}")
    print(f"Requests per second: {requests_per_second:.2f}")


if __name__ == "__main__":
    main()
