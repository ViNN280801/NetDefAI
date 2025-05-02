import time
import random
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor

# Target settings
BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "secret"

# Load XSS patterns from the file
try:
    with open("patterns/xss.txt", "r") as f:
        XSS_PATTERNS = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
    if not XSS_PATTERNS:
        print("Warning: No XSS patterns loaded from patterns/xss.txt")
except FileNotFoundError:
    print("Error: patterns/xss.txt not found. Please ensure the file exists.")
    XSS_PATTERNS = ["<script>alert(1)</script>"]  # Default fallback

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


def generate_xss_event():
    """Generate a security event with XSS payload."""
    source_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    xss_payload = random.choice(XSS_PATTERNS)

    # Inject XSS into a plausible field like 'message' or 'comment'
    return {
        "source_ip": source_ip,
        "event_type": "user_comment",  # Example event type
        "source_type": "XSS Tester",
        "payload": {
            "user": f"user_{random.randint(100,999)}",
            "comment_text": xss_payload,
            "timestamp": time.time(),
        },
    }


def attack_worker(token, requests_count, sleep_min, sleep_max, worker_id):
    """Worker function that sends XSS requests."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    success = 0
    blocked = 0
    failed = 0  # Other failures (timeout, server error, etc.)

    print(f"Worker {worker_id} started - sending {requests_count} XSS requests")

    for i in range(requests_count):
        try:
            event = generate_xss_event()
            response = requests.post(
                f"{BASE_URL}/events", headers=headers, json=event, timeout=5
            )

            if response.status_code == 202:  # Accepted
                success += 1
            elif response.status_code == 403:  # Forbidden (Blocked by signature check)
                blocked += 1
            elif (
                response.status_code == 429
            ):  # Too Many Requests (Blocked by rate limit/error threshold)
                blocked += 1  # Count as blocked for simplicity
                print(f"Worker {worker_id} hit rate limit/error block (status 429)")
            else:
                failed += 1
                print(
                    f"Worker {worker_id} request failed with status {response.status_code}: {response.text[:100]}..."
                )

            sleep_time = random.uniform(sleep_min, sleep_max)
            time.sleep(sleep_time)

        except requests.exceptions.RequestException as e:
            failed += 1
            print(f"Worker {worker_id} error sending request: {e}")
        except Exception as e:
            failed += 1
            print(f"Worker {worker_id} unexpected error: {e}")

    print(
        f"Worker {worker_id} finished - Success: {success}, Blocked: {blocked}, Failed: {failed}"
    )
    return success, blocked, failed


def main():
    parser = argparse.ArgumentParser(description="XSS Test Tool for API Gateway")
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

    print("=== XSS Test Tool for API Gateway ===")
    print(f"Target: {BASE_URL}")
    print(f"Threads: {args.threads}")
    print(f"Requests per thread: {args.requests}")
    print(f"Total XSS patterns loaded: {len(XSS_PATTERNS)}")

    print("Getting authentication token...")
    token = get_token()
    if not token:
        print("Failed to get token. Exiting.")
        return

    print("Starting XSS test...")
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
            try:
                success, blocked, failed = future.result()
                total_success += success
                total_blocked += blocked
                total_failed += failed
            except Exception as e:
                print(f"Error getting result from worker: {e}")
                total_failed += args.requests  # Assume all failed for this worker

    elapsed_time = time.time() - start_time
    total_sent = total_success + total_blocked + total_failed
    requests_per_second = total_sent / elapsed_time if elapsed_time > 0 else 0

    print("=== XSS Test Results ===")
    print(f"Test duration: {elapsed_time:.2f} seconds")
    print(f"Total requests sent (approx): {total_sent}")
    print(f"Successful requests (202 Accepted): {total_success}")
    print(f"Blocked requests (403 Forbidden / 429 Too Many): {total_blocked}")
    print(f"Failed requests (Errors/Other): {total_failed}")
    print(f"Requests per second: {requests_per_second:.2f}")


if __name__ == "__main__":
    main()
