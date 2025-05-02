import time
import random
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor

# Target settings
BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "secret"

# Attack parameters
DEFAULT_THREADS = 10
DEFAULT_REQUESTS_PER_THREAD = 100
DEFAULT_SLEEP_MIN = 0.01
DEFAULT_SLEEP_MAX = 0.1


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


def generate_random_ip():
    """Generate random IP address."""
    return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"


def generate_random_event():
    """Generate random security event."""
    source_ip = generate_random_ip()

    # Generate some variations of events
    event_types = ["connection", "auth", "data_transfer", "scan"]
    source_types = ["Firewall", "IDS", "WAF", "NetworkMonitor"]
    protocols = ["TCP", "UDP", "ICMP", "HTTP"]
    flags = ["SYN", "ACK", "FIN", "RST", "SYN-ACK"]

    return {
        "source_ip": source_ip,
        "destination_ip": generate_random_ip(),
        "event_type": random.choice(event_types),
        "source_type": random.choice(source_types),
        "payload": {
            "protocol": random.choice(protocols),
            "port": random.randint(1, 65535),
            "flags": random.choice(flags),
            "packet_count": random.randint(1, 100),
            "severity": random.randint(1, 10),
            "attack_signature": f"SIG-{random.randint(1000, 9999)}",
        },
    }


def attack_worker(token, requests_count, sleep_min, sleep_max, worker_id):
    """Worker function that sends multiple requests."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    success = 0
    failed = 0

    print(f"Worker {worker_id} started - sending {requests_count} requests")

    for i in range(requests_count):
        try:
            event = generate_random_event()
            response = requests.post(
                f"{BASE_URL}/events", headers=headers, json=event, timeout=5
            )

            if response.status_code == 202:
                success += 1
            else:
                failed += 1
                print(
                    f"Request failed with status {response.status_code}: {response.text}"
                )

            # Random sleep to make the attack more realistic
            sleep_time = random.uniform(sleep_min, sleep_max)
            time.sleep(sleep_time)

        except requests.exceptions.RequestException as e:
            failed += 1
            print(f"Error sending request: {e}")

    print(f"Worker {worker_id} finished - Success: {success}, Failed: {failed}")
    return success, failed


def main():
    parser = argparse.ArgumentParser(description="DDoS Test Tool for API Gateway")
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
        help=f"Minimum sleep between requests in seconds (default: {DEFAULT_SLEEP_MIN})",
    )
    parser.add_argument(
        "-m",
        "--sleep-max",
        type=float,
        default=DEFAULT_SLEEP_MAX,
        help=f"Maximum sleep between requests in seconds (default: {DEFAULT_SLEEP_MAX})",
    )
    args = parser.parse_args()

    print("=== DDoS Test Tool for API Gateway ===")
    print(f"Target: {BASE_URL}")
    print(f"Threads: {args.threads}")
    print(f"Requests per thread: {args.requests}")
    print(f"Sleep between requests: {args.sleep_min}-{args.sleep_max} seconds")
    print(f"Total requests to be sent: {args.threads * args.requests}")

    print("\nGetting authentication token...")
    token = get_token()
    if not token:
        print("Failed to get token. Exiting.")
        return

    print("Starting DDoS test...")
    start_time = time.time()

    total_success = 0
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
            success, failed = future.result()
            total_success += success
            total_failed += failed

    elapsed_time = time.time() - start_time
    requests_per_second = (total_success + total_failed) / elapsed_time

    print("\n=== DDoS Test Results ===")
    print(f"Test duration: {elapsed_time:.2f} seconds")
    print(f"Successful requests: {total_success}")
    print(f"Failed requests: {total_failed}")
    print(f"Requests per second: {requests_per_second:.2f}")


if __name__ == "__main__":
    main()
