# Web Attack Detection System

A machine learning-based threat detection system for distributed applications that leverages deep learning models to identify various web attacks.

## Overview

This system can detect different types of web attacks:

- SQL Injections
- Cross-Site Scripting (XSS) 
- Path Traversal
- Denial of Service (DoS)

The system uses models trained on synthetic datasets generated from attack patterns in the `patterns` directory.

## Getting Started

### Requirements

- OS: Windows
- Python 3.12.7 (CUDA/Tensorflow supported) or higher (CUDA/Tensorflow not supported)
- PyTorch
- TensorFlow
- scikit-learn
- NumPy
- Pandas

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Generating Datasets

To generate datasets for all attack types:

```bash
python generate_dataset.py --attack-type all --num-samples 5000 --malicious-ratio 0.5
```

To generate a dataset for a specific attack type:

```bash
python generate_dataset.py --attack-type sql_injection --num-samples 5000 --malicious-ratio 0.5
```

### Training Models

To train models for all attack types:

```bash
python train_all_models.py --num-samples 5000 --malicious-ratio 0.5 --device auto
```

To train a model for a specific attack type:

```bash
python train_all_models.py --attack-type xss --num-samples 5000 --malicious-ratio 0.5 --device auto
```

## Model Training Details

The system uses a universal trainer that supports multiple model types:

- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
- Multi-Layer Perceptron (MLP)
- Artificial Neural Network (ANN) with PyTorch

By default, the system uses ANN models with TF-IDF vectorization, which have shown the best performance for detecting web attacks.

### Hardware Acceleration

The training system automatically detects and uses available hardware acceleration:

- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- Multi-core CPU support

## Adding New Attack Patterns

1. Add new patterns to the appropriate file in the `patterns/` directory
2. Run the dataset generation and model training scripts

# Threat Analysis System

A microservices-based system for detecting and analyzing security threats using AI.

## System Components

- **API Gateway**: Receives security events and authentication
- **AI Analyzer**: Processes events to detect threats
- **Alert Service**: Sends notifications for detected threats
- **RabbitMQ**: Message queue for event processing
- **Cassandra**: Long-term storage for security events
- **Redis**: Cache for fast access to detection rules and results
- **Prometheus/Grafana**: Monitoring and visualization

## Running the Application

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for test scripts)
- Requests library (`pip install requests`)

### Starting the Services

```powershell
# Clone the repository
# cd to project directory

# Start all services
docker-compose up -d

# Check if all services are running
docker-compose ps
```

The services will be available at:
- API Gateway: http://localhost:8000
- RabbitMQ Management: http://localhost:15672 (guest/guest)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Basic API Usage

1. Get an authentication token:
```
POST http://localhost:8000/token
Content-Type: application/x-www-form-urlencoded

username=admin&password=secret
```

2. Use this token for all subsequent requests:
```
POST http://localhost:8000/events
Authorization: Bearer <your_token>
Content-Type: application/json

{
  "source_ip": "192.168.1.100",
  "destination_ip": "192.168.1.5", 
  "event_type": "connection",
  "source_type": "Firewall",
  "payload": {
    "protocol": "TCP",
    "port": 443,
    "flags": "SYN",
    "packet_count": 5
  }
}
```

## Testing Scripts

This repository includes three Python scripts for testing:

### 1. Basic API Testing

```powershell
pip install requests
python test_api.py
```

This script authenticates with the API and sends a test security event.

### 2. DDoS Attack Simulation

```powershell
# Run with default settings (10 threads, 100 requests each)
python ddos_test.py

# Or customize parameters
python ddos_test.py --threads 20 --requests 200 --sleep-min 0.005 --sleep-max 0.05
```

This simulates a DDoS attack by sending many security events from multiple threads.

### 3. Check for DDoS Detection

```powershell
# After running the DDoS test, check if it was detected
python check_alerts.py
```

This script checks system metrics to see if the DDoS attack was detected and if any countermeasures were activated.

## Monitoring the System

During and after testing, you can:

1. Check Prometheus for real-time metrics (http://localhost:9090)
2. View Grafana dashboards for visualizations (http://localhost:3000)
3. Examine RabbitMQ queues to see message processing (http://localhost:15672)

## Stopping the Services

```powershell
# Stop all services
docker-compose down

# Remove volumes too if needed
docker-compose down -v
```
