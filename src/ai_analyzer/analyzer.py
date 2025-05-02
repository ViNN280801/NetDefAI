import os
import json
import time
import pika
import torch
import joblib
import signal
import threading
import collections
import numpy as np
from torch import nn
from torch.optim import Adam
from typing import Dict, Any
from datetime import datetime
from prometheus_client import Counter, Gauge
from src.logger import analyzer_logger as logger
from src.config.yaml_config_loader import YamlConfigLoader
from src.model_training.device_manager import get_device_manager, ComputeDevice
from src.utils.db_connectors import CassandraConnector, RedisConnector


# Load configuration
config_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config.yaml",
)
config = YamlConfigLoader(config_path)


class AIAnalyzer:
    """
    AI Analyzer service that processes security events from RabbitMQ queue
    and applies AI/ML models to detect threats.
    """

    def __init__(self):
        """Initialize the AI Analyzer with configuration and connections."""
        logger.info("Initializing AI Analyzer service")

        # JSON serialization helper
        self.json_serializer = lambda obj: (
            obj.isoformat() if isinstance(obj, datetime) else None
        )

        # RabbitMQ configuration
        self.rabbitmq_host = config.get("rabbitmq.host", "localhost")
        self.rabbitmq_port = config.get("rabbitmq.port", 5672)
        self.rabbitmq_user = config.get("rabbitmq.user", "guest")
        self.rabbitmq_pass = config.get("rabbitmq.password", "guest")
        self.input_queue = config.get("rabbitmq.queues.threat_events", "threat_events")
        self.output_queue = config.get("rabbitmq.queues.alerts", "alerts")

        # Database configuration
        self.cassandra_enabled = config.get("cassandra.enabled", True)
        self.redis_enabled = config.get("redis.enabled", True)
        self.cassandra = None
        self.redis = None

        # DDoS protection settings
        self.error_threshold = config.get(
            "ai_analyzer.error_threshold", 5
        )  # Number of errors before blocking
        self.error_window = config.get(
            "ai_analyzer.error_window", 60
        )  # Time window in seconds
        self.block_duration = config.get(
            "ai_analyzer.block_duration", 300
        )  # Block duration in seconds
        self.ip_error_counters = {}  # Track errors by IP
        self.blocked_ips = {}  # Track blocked IPs and their expiry time

        # Model configuration
        self.model_path = config.get(
            "ai_analyzer.model_path", "../models/unified_model_pytorch_model.pt"
        )
        self.joblib_model_path = config.get(
            "ai_analyzer.joblib_model_path", "../models/unified_model.joblib"
        )
        self.threshold = config.get(
            "ai_analyzer.threshold", 0.7
        )  # Confidence threshold

        # Continuous learning configuration
        self.enable_continuous_learning = config.get(
            "ai_analyzer.enable_continuous_learning", True
        )
        self.training_buffer_size = config.get("ai_analyzer.training_buffer_size", 1000)
        self.training_interval = config.get(
            "ai_analyzer.training_interval", 3600
        )  # seconds
        self.min_samples_to_train = config.get("ai_analyzer.min_samples_to_train", 100)
        self.learning_rate = config.get("ai_analyzer.learning_rate", 0.001)
        self.training_epochs = config.get("ai_analyzer.training_epochs", 5)

        # Buffers for training data
        self.training_buffer = []  # List of (features, label) tuples
        self.last_training_time = time.time()

        # Training metrics
        self.model_updates_counter = Counter(
            "model_updates_total", "Total number of model updates"
        )
        self.model_accuracy_gauge = Gauge(
            "model_accuracy", "Current model accuracy on validation data"
        )

        # Thread control
        self.should_exit = threading.Event()
        self.connection = None
        self.channel = None

        # Setup compute device for model inference
        self.device_manager = get_device_manager()
        preferred_device = config.get("ai_analyzer.preferred_device", None)
        if preferred_device:
            preferred_device = ComputeDevice[preferred_device]
        self.device = self.device_manager.select_device(preferred_device)

        # Set PyTorch device
        if self.device == ComputeDevice.CUDA and torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
        elif (
            self.device == ComputeDevice.MPS
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            self.torch_device = torch.device("mps")
        else:
            self.torch_device = torch.device("cpu")

        logger.info(
            f"Using compute device: {self.device}, PyTorch device: {self.torch_device}"
        )

        # Initialize database connections
        if self.cassandra_enabled:
            try:
                self.cassandra = CassandraConnector()
                logger.info("Connected to Cassandra database")
            except Exception as e:
                logger.error(f"Failed to connect to Cassandra: {str(e)}")
                self.cassandra_enabled = False
                logger.warning("Continuing without Cassandra support")

        if self.redis_enabled:
            try:
                self.redis = RedisConnector()
                logger.info("Connected to Redis database")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                self.redis_enabled = False
                logger.warning("Continuing without Redis support")

        # Load ML model - this is required
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Critical error loading model: {str(e)}")
            logger.warning(
                "Attempting to create a dummy model for minimal functionality"
            )
            # Create a minimal dummy model
            try:
                self.model = nn.Sequential(
                    nn.Linear(100, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                ).to(self.torch_device)
                self.model.eval()
                self.model_type = "pytorch_dummy"
                logger.info("Initialized dummy PyTorch model")
            except Exception as e2:
                logger.error(f"Failed to create dummy model: {str(e2)}")
                raise

        # Start continuous learning thread if enabled
        if self.enable_continuous_learning:
            self.training_thread = threading.Thread(
                target=self._continuous_learning_loop
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            logger.info("Continuous learning thread started")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start cleanup thread for blocked IPs
        self.cleanup_thread = threading.Thread(target=self._cleanup_blocked_ips)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()

        # Add metrics
        self.blocked_ips_counter = Counter(
            "blocked_ips_total", "Total number of blocked IPs", ["ip"]
        )
        self.current_blocked_ips_gauge = Gauge(
            "current_blocked_ips", "Current number of blocked IPs"
        )
        self.sqli_attempts_counter = Counter(
            "sqli_attempts_total",
            "Total number of detected SQL injection attempts",
            ["source_ip"],
        )
        self.current_sqli_ips_gauge = Gauge(
            "current_sqli_ips", "Current number of unique IPs with SQLi attempts"
        )

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _load_model(self):
        """Load the trained ML model for threat detection."""
        try:
            # Try to load the PyTorch model
            if os.path.exists(self.model_path):
                logger.info(f"Loading PyTorch model from {self.model_path}")
                loaded_obj = torch.load(self.model_path, map_location=self.torch_device)

                # Check if the loaded object is a state_dict (OrderedDict)
                if isinstance(loaded_obj, dict) or isinstance(
                    loaded_obj, collections.OrderedDict
                ):
                    self.model = nn.Sequential(
                        nn.Linear(100, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid(),
                    ).to(self.torch_device)

                    # Try to load the state dict
                    try:
                        self.model.load_state_dict(loaded_obj)
                        logger.info(
                            "Loaded state_dict into placeholder model successfully"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not load state_dict into placeholder model: {str(e)}"
                        )
                else:
                    # Loaded object is a full model
                    self.model = loaded_obj

                self.model.eval()  # Set model to evaluation mode
                self.model_type = "pytorch"
                logger.info("PyTorch model loaded successfully")
            # Try to load the joblib model as fallback
            elif os.path.exists(self.joblib_model_path):
                logger.info(f"Loading joblib model from {self.joblib_model_path}")
                self.model = joblib.load(self.joblib_model_path)
                self.model_type = "joblib"
                logger.info("Joblib model loaded successfully")
            else:
                raise FileNotFoundError("No model file found at the specified paths")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _save_model(self):
        """Save the current model to file."""
        try:
            # Create models directory if it doesn't exist
            model_dir = os.path.dirname(self.model_path)
            os.makedirs(model_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Handle PyTorch models
            if self.model_type in ["pytorch", "pytorch_dummy"]:
                # Check if it's really a PyTorch model
                if not isinstance(self.model, nn.Module):
                    logger.error(
                        "Model is marked as PyTorch but is not a nn.Module instance"
                    )
                    return False

                # Save main model file
                try:
                    torch.save(self.model, self.model_path)
                    logger.info(f"Saved PyTorch model to {self.model_path}")

                    # Save backup
                    backup_path = self.model_path.replace(
                        ".pt", f"_backup_{timestamp}.pt"
                    )
                    torch.save(self.model, backup_path)
                    logger.info(f"Created PyTorch model backup at {backup_path}")
                except Exception as e:
                    logger.error(f"Failed to save PyTorch model: {str(e)}")
                    return False

            # Handle joblib models
            elif self.model_type == "joblib":
                try:
                    # Save main model file
                    joblib.dump(self.model, self.joblib_model_path)
                    logger.info(f"Saved joblib model to {self.joblib_model_path}")

                    # Save backup
                    backup_path = self.joblib_model_path.replace(
                        ".joblib", f"_backup_{timestamp}.joblib"
                    )
                    joblib.dump(self.model, backup_path)
                    logger.info(f"Created joblib model backup at {backup_path}")
                except Exception as e:
                    logger.error(f"Failed to save joblib model: {str(e)}")
                    return False
            else:
                logger.error(f"Unknown model type {self.model_type}, cannot save")
                return False

            return True
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False

    def _connect_rabbitmq(self):
        """Establish connection to RabbitMQ."""
        try:
            # Setup RabbitMQ connection
            credentials = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_pass)
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Declare queues
            self.channel.queue_declare(queue=self.input_queue, durable=True)
            self.channel.queue_declare(queue=self.output_queue, durable=True)

            # Set QoS prefetch to control load
            self.channel.basic_qos(prefetch_count=1)

            logger.info(
                f"Connected to RabbitMQ at {self.rabbitmq_host}:{self.rabbitmq_port}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            return False

    def _preprocess_event(self, event: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess an event for model input.

        Args:
            event: The event data dictionary

        Returns:
            Preprocessed features as numpy array
        """
        # Extract relevant features from the event
        features = []

        # Add source_ip based features (convert to numerical representation)
        if "source_ip" in event:
            ip_parts = event["source_ip"].split(".")
            if len(ip_parts) == 4:
                for part in ip_parts:
                    try:
                        features.append(int(part) / 255.0)  # Normalize IP components
                    except ValueError:
                        features.append(0.0)
            else:
                features.extend([0.0] * 4)
        else:
            features.extend([0.0] * 4)

        # Add event_type as one-hot encoding (simplified)
        event_types = ["connection", "auth", "data", "command", "other"]
        event_type = event.get("event_type", "other")
        for e_type in event_types:
            features.append(1.0 if event_type == e_type else 0.0)

        # Add more features based on payload
        payload = event.get("payload", {})

        # Example: Extract protocol information
        protocol = payload.get("protocol", "unknown")
        protocols = ["tcp", "udp", "http", "https", "dns", "unknown"]
        for p in protocols:
            features.append(1.0 if protocol == p else 0.0)

        # Example: port numbers (normalized)
        src_port = payload.get("src_port", 0)
        dst_port = payload.get("dst_port", 0)
        features.append(min(src_port, 65535) / 65535.0)
        features.append(min(dst_port, 65535) / 65535.0)

        # Example: packet size (normalized)
        packet_size = payload.get("size", 0)
        features.append(min(packet_size, 1500) / 1500.0)  # Normalize by typical MTU

        # Pad the feature vector to 100 dimensions with zeros
        padded_features = np.zeros(100, dtype=np.float32)
        padded_features[: len(features)] = (
            features  # Fill the first positions with actual data
        )
        # and the rest with zeros

        return padded_features

    def _analyze_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an event using the loaded model.

        Note: Immediate signature-based SQLi checks are now handled by the API Gateway.
        This focuses on ML-based detection.
        """
        try:
            features = self._preprocess_event(event)
            confidence = 0.0

            # PyTorch model prediction
            if self.model_type in ["pytorch", "pytorch_dummy"]:
                if isinstance(self.model, nn.Module):
                    # Convert numpy array to torch tensor
                    tensor_features = torch.tensor(features, dtype=torch.float32).to(
                        self.torch_device
                    )

                    # Get prediction using PyTorch model
                    with torch.no_grad():
                        # Ensure the input is properly shaped for the model
                        input_tensor = tensor_features.unsqueeze(0)
                        output = self.model(input_tensor)

                        # Extract the confidence value
                        if isinstance(output, torch.Tensor):
                            confidence = float(output.item())
                        else:
                            confidence = 0.5  # Fallback
                            logger.warning(
                                "Model output is not a tensor, using default confidence"
                            )
                else:
                    logger.error(
                        "Model is marked as PyTorch but is not a nn.Module instance"
                    )
                    confidence = 0.5  # Use a middle value as fallback

            # scikit-learn model prediction
            elif self.model_type == "joblib":
                # Convert to the right format for scikit-learn
                features_reshaped = features.reshape(1, -1)

                # Try predict_proba first (most common for classifiers)
                if hasattr(self.model, "predict_proba"):
                    try:
                        # For classifiers that support probability estimates
                        probabilities = self.model.predict_proba(features_reshaped)
                        if (
                            probabilities.shape[1] >= 2
                        ):  # Binary or multi-class classification
                            # Use the probability of the positive class (typically index 1)
                            confidence = float(probabilities[0][1])
                        else:
                            # Only one class in probabilities, use it directly
                            confidence = float(probabilities[0][0])
                    except Exception as e:
                        logger.error(f"Error in predict_proba: {str(e)}")
                        confidence = 0.5

                # Try decision_function as fallback
                elif hasattr(self.model, "decision_function"):
                    try:
                        # For models like SVM that provide a decision function
                        decision_values = self.model.decision_function(
                            features_reshaped
                        )

                        # Convert to a probability-like value with sigmoid
                        if isinstance(decision_values, np.ndarray):
                            if decision_values.size > 0:
                                # Convert raw decision value to [0,1] range with sigmoid function
                                confidence = float(
                                    1.0 / (1.0 + np.exp(-decision_values[0]))
                                )
                            else:
                                confidence = 0.5
                        else:
                            # If it's a scalar value
                            confidence = float(1.0 / (1.0 + np.exp(-decision_values)))
                    except Exception as e:
                        logger.error(f"Error in decision_function: {str(e)}")
                        confidence = 0.5

                # Last resort: use predict
                else:
                    try:
                        prediction = self.model.predict(features_reshaped)
                        # Assuming binary 0/1 prediction
                        if isinstance(prediction, np.ndarray) and prediction.size > 0:
                            confidence = float(prediction[0])
                        else:
                            confidence = float(prediction)
                    except Exception as e:
                        logger.error(f"Error in predict: {str(e)}")
                        confidence = 0.5

            # Unknown model type
            else:
                logger.warning(f"Unknown model type: {self.model_type}")
                confidence = 0.5

            # Ensure confidence is a valid float between 0 and 1
            confidence = max(0.0, min(1.0, float(confidence)))

            # Determine if it's a threat based on confidence threshold
            is_threat = confidence >= self.threshold

            # Store for incremental training if continuous learning is enabled
            if self.enable_continuous_learning:
                self._add_to_training_buffer(features, is_threat)

            # Create the analysis result
            result = {
                "event_id": event.get("id", str(time.time())),
                "is_threat": is_threat,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "source_event": event,
                "details": {
                    "model_type": self.model_type,
                    "threshold": self.threshold,
                    "features": features.tolist(),
                },
            }

            # Determine threat type if applicable
            if is_threat:
                # This would be more sophisticated in a real system
                if "auth" in event.get("event_type", ""):
                    result["threat_type"] = "authentication_attack"
                elif confidence > 0.9:
                    result["threat_type"] = "severe_attack"
                else:
                    result["threat_type"] = "suspicious_activity"

            if result.get("threat_type") == "sql_injection":
                self.sqli_attempts_counter.labels(
                    source_ip=event.get("source_ip", "unknown")
                ).inc()
                if self.redis_enabled and self.redis is not None:
                    try:
                        self.redis.add_to_set(
                            "sqli_ips", event.get("source_ip", "unknown")
                        )
                    except Exception as e:
                        logger.error(f"Failed to store SQLi IP in Redis: {str(e)}")
                self.current_sqli_ips_gauge.set(len(self._get_unique_sqli_ips()))

            return result
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            if "source_ip" in event:
                self._track_error(event["source_ip"], str(e))
            return {
                "event_id": event.get("id", str(time.time())),
                "is_threat": False,
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _store_result(self, result: Dict[str, Any]):
        """
        Store the analysis result in databases.

        Args:
            result: The analysis result
        """
        # Store in Cassandra for permanent storage
        if self.cassandra_enabled and self.cassandra is not None:
            try:
                # Simplified - in reality, would use proper schema
                event_id = result["event_id"]
                is_threat = result["is_threat"]
                confidence = result["confidence"]
                timestamp = result["timestamp"]
                source_ip = result.get("source_event", {}).get("source_ip", "unknown")
                threat_type = result.get("threat_type", "none")

                self.cassandra.insert_event(
                    event_id=event_id,
                    timestamp=timestamp,
                    source_ip=source_ip,
                    is_threat=is_threat,
                    confidence=confidence,
                    threat_type=threat_type,
                    details=json.dumps(result, default=self.json_serializer),
                )
                logger.info(f"Stored result for event {event_id} in Cassandra")
            except Exception as e:
                logger.error(f"Failed to store result in Cassandra: {str(e)}")

        # Cache in Redis for quick access
        if (
            self.redis_enabled
            and self.redis is not None
            and result.get("is_threat", False)
        ):
            try:
                event_id = result["event_id"]
                ttl = 3600  # Cache for 1 hour

                # Store the result - handle datetime objects with our serializer
                self.redis.set_json(f"threat:{event_id}", result, ttl)

                # Add to recent threats list
                # Convert the timestamp string back to datetime if needed
                try:
                    if isinstance(result["timestamp"], str):
                        timestamp_value = float(
                            datetime.fromisoformat(result["timestamp"]).timestamp()
                        )
                    else:
                        timestamp_value = float(datetime.now().timestamp())
                except Exception:
                    timestamp_value = float(time.time())

                self.redis.add_to_sorted_set(
                    "recent_threats",
                    event_id,
                    timestamp_value,
                    max_size=100,
                )

                # If source IP is available, add to threat sources set
                source_ip = result.get("source_event", {}).get("source_ip")
                if source_ip:
                    self.redis.add_to_set("threat_sources", source_ip)

                logger.info(f"Cached threat event {event_id} in Redis")
            except Exception as e:
                logger.error(f"Failed to cache result in Redis: {str(e)}")

    def _is_ip_blocked(self, ip_address):
        """
        Check if an IP address is currently blocked.

        Args:
            ip_address: The IP address to check

        Returns:
            bool: True if the IP is blocked, False otherwise
        """
        if ip_address in self.blocked_ips:
            if time.time() < self.blocked_ips[ip_address]:
                return True
            else:
                # Remove expired block
                del self.blocked_ips[ip_address]
        return False

    def _track_error(self, ip_address, error_message):
        """Track errors and block IPs with abnormal activity."""
        current_time = time.time()

        if ip_address not in self.ip_error_counters:
            self.ip_error_counters[ip_address] = []

        # Remove old errors
        self.ip_error_counters[ip_address] = [
            t
            for t in self.ip_error_counters[ip_address]
            if current_time - t < self.error_window
        ]

        # Add current error
        self.ip_error_counters[ip_address].append(current_time)

        # Block if threshold exceeded
        if len(self.ip_error_counters[ip_address]) >= self.error_threshold:
            expiry_time = current_time + self.block_duration
            self.blocked_ips[ip_address] = expiry_time
            self.ip_error_counters[ip_address] = []
            logger.warning(
                "BLOCKED IP: "
                f"{ip_address} for {self.block_duration}s"
                f"(errors: {self.error_threshold}/{self.error_window}s)"
            )

            # Sync block with Redis (if enabled)
            if self.redis_enabled and self.redis is not None:
                self.redis.set(
                    f"blocked_ip:{ip_address}", "1", int(self.block_duration)
                )
                self.redis.add_to_set("blocked_ips", ip_address)

            # Update metrics
            self.blocked_ips_counter.labels(ip=ip_address).inc()
            self.current_blocked_ips_gauge.set(len(self.blocked_ips))

            return True
        return False

    def _process_message(self, ch, method, properties, body):
        """
        Process a message from the queue.

        Args:
            ch: Channel object
            method: Method frame
            properties: Properties
            body: Message body
        """
        try:
            # Parse the message
            message = json.loads(body.decode())
            event_id = message.get("id", "unknown")
            source_ip = message.get("source_ip", "unknown")

            # Check if this IP is blocked
            if self._is_ip_blocked(source_ip):
                logger.warning(f"Rejecting message from blocked IP: {source_ip}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            logger.info(f"Processing event {event_id} from {source_ip}")

            # Analyze the event
            result = self._analyze_event(message)

            # Store the result
            self._store_result(result)

            # Publish an alert if it's a threat
            if result.get("is_threat", False):
                self._publish_alert(result)
                logger.info(
                    f"Event {event_id} identified as threat with confidence {result['confidence']}"
                )
            else:
                logger.info(
                    f"Event {event_id} is not a threat (confidence: {result['confidence']})"
                )

            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except json.JSONDecodeError:
            logger.error("Received invalid JSON message")
            # Try to extract the source IP for tracking
            try:
                decoded = body.decode()
                if '"source_ip"' in decoded:
                    import re

                    match = re.search(r'"source_ip"\s*:\s*"([^"]+)"', decoded)
                    if match:
                        source_ip = match.group(1)
                        self._track_error(source_ip, "Invalid JSON")
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing message: {error_msg}")

            # Check for datetime serialization errors
            if "Object of type datetime is not JSON serializable" in error_msg:
                try:
                    message = json.loads(body.decode())
                    source_ip = message.get("source_ip", "unknown")

                    # Track error and check if IP should be blocked
                    if self._track_error(source_ip, error_msg):
                        # IP was blocked, don't requeue
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                        return
                except Exception:
                    # Failed to extract IP, continue normal processing
                    pass

            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def _publish_alert(self, result: Dict[str, Any]):
        """
        Publish an alert to the alert queue if the event is a threat.

        Args:
            result: The analysis result
        """
        if not result.get("is_threat", False):
            return

        try:
            # Create alert message
            alert = {
                "event_id": result["event_id"],
                "timestamp": datetime.now().isoformat(),
                "threat_type": result.get("threat_type", "unknown"),
                "confidence": result["confidence"],
                "source_ip": result.get("source_event", {}).get("source_ip", "unknown"),
                "details": result,
            }

            # Publish to RabbitMQ if channel is available
            if self.channel:
                try:
                    self.channel.basic_publish(
                        exchange="",
                        routing_key=self.output_queue,
                        body=json.dumps(alert, default=self.json_serializer).encode(),
                        properties=pika.BasicProperties(
                            delivery_mode=2,  # Make message persistent
                            content_type="application/json",
                        ),
                    )
                    logger.info(
                        f"Published alert for event {result['event_id']} to queue {self.output_queue}"
                    )
                except Exception as e:
                    logger.error(f"Failed to publish to RabbitMQ: {str(e)}")
                    # Log the alert details for backup
                    logger.warning(
                        f"Alert that couldn't be sent: {json.dumps(alert, default=self.json_serializer)}"
                    )
            else:
                logger.warning("RabbitMQ channel not initialized, can't publish alert")
                # Log the alert details for backup
                logger.warning(
                    f"Alert that couldn't be sent: {json.dumps(alert, default=self.json_serializer)}"
                )
        except Exception as e:
            logger.error(f"Failed to publish alert: {str(e)}")

    def start(self):
        """Start the AI Analyzer service."""
        logger.info("Starting AI Analyzer service")

        # Connect to RabbitMQ
        retry_count = 0
        max_retries = 5
        connected = False

        while not connected and retry_count < max_retries:
            connected = self._connect_rabbitmq()
            if not connected:
                retry_count += 1
                sleep_time = min(30, 2**retry_count)  # Exponential backoff
                logger.info(
                    f"Retrying RabbitMQ connection in {sleep_time} seconds (attempt {retry_count}/{max_retries})"
                )
                time.sleep(sleep_time)

        if not connected:
            logger.error("Failed to connect to RabbitMQ after multiple attempts")
            logger.warning(
                "AI Analyzer will run in limited mode (no message processing)"
            )
            return

        # Start consuming messages
        if not self.channel:
            msg = "RabbitMQ channel not initialized"
            logger.error(msg)
            logger.warning(
                "AI Analyzer will run in limited mode (no message processing)"
            )
            return

        try:
            self.channel.basic_consume(
                queue=self.input_queue, on_message_callback=self._process_message
            )

            logger.info(f"Waiting for messages on queue {self.input_queue}")

            # Start consumption loop
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Error in consumption loop: {str(e)}")
            self.stop()

    def stop(self):
        """Stop the AI Analyzer service."""
        logger.info("Stopping AI Analyzer service")

        # Save the model before shutting down
        if (
            self.enable_continuous_learning
            and len(self.training_buffer) > self.min_samples_to_train // 2
        ):
            logger.info("Performing final model training before shutdown")
            try:
                self._train_model_incremental()
            except Exception as e:
                logger.error(f"Error during final training: {str(e)}")

        # Signal threads to exit
        self.should_exit.set()

        # Stop consuming
        if self.channel:
            try:
                self.channel.stop_consuming()
            except Exception as e:
                logger.error(f"Error stopping consumption: {str(e)}")

        # Close connection
        if self.connection and self.connection.is_open:
            try:
                self.connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")

        # Wait for cleanup thread
        if hasattr(self, "cleanup_thread") and self.cleanup_thread.is_alive():
            try:
                self.cleanup_thread.join(
                    timeout=2.0
                )  # Allow 2 seconds for thread to terminate
            except Exception as e:
                logger.error(f"Error joining cleanup thread: {str(e)}")

        # Close database connections
        if self.redis is not None:
            try:
                self.redis.close()
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")

        if self.cassandra is not None:
            try:
                self.cassandra.close()
            except Exception as e:
                logger.error(f"Error closing Cassandra connection: {str(e)}")

        logger.info("AI Analyzer service stopped")

    def _cleanup_blocked_ips(self):
        """Cleanup blocked IPs and remove expired entries."""
        while not self.should_exit.is_set():
            try:
                current_time = time.time()
                expired_ips = [
                    ip
                    for ip, expiry in self.blocked_ips.items()
                    if expiry <= current_time
                ]
                for ip in expired_ips:
                    del self.blocked_ips[ip]
                    logger.info(f"Unblocked IP: {ip}")
                time.sleep(self.block_duration)
            except Exception as e:
                logger.error(f"Error in cleanup thread: {str(e)}")
                time.sleep(self.block_duration)

    def _get_unique_sqli_ips(self) -> set:
        """Returns a set of unique IPs with SQLi attempts."""
        if not self.redis_enabled or self.redis is None:
            return set()

        try:
            return set(self.redis.get_set_members("sqli_ips"))
        except Exception as e:
            logger.error(f"Failed to fetch SQLi IPs from Redis: {str(e)}")
            return set()

    def _add_to_training_buffer(self, features: np.ndarray, is_threat: bool):
        """Add features and label to training buffer for incremental learning.

        Args:
            features: The feature vector
            is_threat: Whether the event was classified as a threat
        """
        try:
            # Convert is_threat to appropriate label format
            label = 1.0 if is_threat else 0.0

            # Add to buffer
            self.training_buffer.append((features, label))

            # Trim buffer if it exceeds the maximum size
            if len(self.training_buffer) > self.training_buffer_size:
                # Remove oldest samples
                self.training_buffer = self.training_buffer[-self.training_buffer_size:]

            # Check if it's time to train
            current_time = time.time()
            if (
                current_time - self.last_training_time > self.training_interval
                and len(self.training_buffer) >= self.min_samples_to_train
            ):
                # Train in a separate thread to avoid blocking
                threading.Thread(target=self._train_model_incremental).start()
                self.last_training_time = current_time
        except Exception as e:
            logger.error(f"Error adding to training buffer: {str(e)}")

    def _continuous_learning_loop(self):
        """Background thread for continuous model training."""
        while not self.should_exit.is_set():
            try:
                # Sleep for a bit
                time.sleep(10)  # Check every 10 seconds

                # Check if it's time to train
                current_time = time.time()
                if (
                    current_time - self.last_training_time > self.training_interval
                    and len(self.training_buffer) >= self.min_samples_to_train
                ):
                    logger.info("Starting scheduled model retraining")
                    self._train_model_incremental()
                    self.last_training_time = current_time
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {str(e)}")
                time.sleep(60)  # Sleep longer on error

    def _train_model_incremental(self):
        """Train the model with new data from the training buffer."""
        if not self.training_buffer:
            logger.info("No training data available, skipping training")
            return

        logger.info(
            f"Starting incremental training with {len(self.training_buffer)} samples"
        )

        try:
            # Extract features and labels from the buffer
            features_list = []
            labels_list = []

            for features, label in self.training_buffer:
                features_list.append(features)
                labels_list.append(label)

            # Convert to numpy arrays
            X = np.array(features_list, dtype=np.float32)
            y = np.array(labels_list, dtype=np.float32)

            # Split data into training and validation sets
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            training_succeeded = False

            # Train based on model type
            if self.model_type in ["pytorch", "pytorch_dummy"]:
                if isinstance(self.model, nn.Module):
                    try:
                        self._train_pytorch_model(X_train, y_train, X_val, y_val)
                        training_succeeded = True
                    except Exception as e:
                        logger.error(f"PyTorch training failed: {str(e)}")
                else:
                    logger.error("Cannot train: model is not a PyTorch nn.Module")

            elif self.model_type == "joblib":
                try:
                    self._train_sklearn_model(X_train, y_train, X_val, y_val)
                    training_succeeded = True
                except Exception as e:
                    logger.error(f"Scikit-learn training failed: {str(e)}")

            else:
                logger.warning(f"Unknown model type {self.model_type}, cannot train")
                return

            # Update training buffer and save model if training succeeded
            if training_succeeded:
                # Reduce buffer size while keeping the most recent data
                self.training_buffer = self.training_buffer[-self.min_samples_to_train:]

                # Save the trained model
                if self._save_model():
                    logger.info("Model saved successfully after training")
                    self.model_updates_counter.inc()

                    # Report metrics
                    try:
                        # This is a simple evaluation for monitoring
                        y_pred = self._predict_for_evaluation(X_val)
                        acc = np.mean(
                            ((y_pred > 0.5).astype(np.float32) == y_val).astype(
                                np.float32
                            )
                        )
                        self.model_accuracy_gauge.set(float(acc))
                        logger.info(f"Validation accuracy after training: {acc:.4f}")
                    except Exception as e:
                        logger.error(f"Failed to compute validation metrics: {str(e)}")

        except Exception as e:
            logger.error(f"Error in incremental training: {str(e)}")

    def _predict_for_evaluation(self, X):
        """Helper method to get predictions for evaluation in a consistent format."""
        if self.model_type in ["pytorch", "pytorch_dummy"] and isinstance(
            self.model, nn.Module
        ):
            # PyTorch prediction
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.torch_device)
                outputs = self.model(X_tensor)
                return outputs.cpu().numpy()

        elif self.model_type == "joblib":
            # scikit-learn prediction
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)
                if probs.shape[1] >= 2:
                    return probs[:, 1]  # Return probability of positive class
                else:
                    return probs[:, 0]
            elif hasattr(self.model, "decision_function"):
                decisions = self.model.decision_function(X)
                # Convert to probabilities using sigmoid
                return 1.0 / (1.0 + np.exp(-decisions))
            else:
                return self.model.predict(X)

        else:
            # Fallback
            return np.random.random(size=len(X))

    def _train_pytorch_model(self, X_train, y_train, X_val, y_val):
        """Train a PyTorch model incrementally."""
        if not isinstance(self.model, nn.Module):
            raise TypeError("Model is not a PyTorch nn.Module")

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(
            self.torch_device
        )
        y_train_tensor = (
            torch.tensor(y_train, dtype=torch.float32)
            .reshape(-1, 1)
            .to(self.torch_device)
        )
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.torch_device)
        y_val_tensor = (
            torch.tensor(y_val, dtype=torch.float32)
            .reshape(-1, 1)
            .to(self.torch_device)
        )

        # Set up training
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCELoss()

        # Training loop
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.training_epochs):
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = loss_fn(outputs, y_train_tensor)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = loss_fn(val_outputs, y_val_tensor)

                # Calculate metrics
                val_preds = (val_outputs > 0.5).float()
                accuracy = (val_preds == y_val_tensor).float().mean().item()

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.training_epochs}, "
                f"Loss: {loss.item():.4f}, "
                f"Val Loss: {val_loss.item():.4f}, "
                f"Accuracy: {accuracy:.4f}"
            )

            # Save best model
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Set back to evaluation mode
        self.model.eval()

    def _train_sklearn_model(self, X_train, y_train, X_val, y_val):
        """Train a scikit-learn model incrementally."""
        from sklearn.ensemble import RandomForestClassifier

        # Check for models that support partial_fit
        if hasattr(self.model, "partial_fit") and callable(self.model.partial_fit):
            try:
                # Models like SGDClassifier that support incremental learning
                unique_classes = np.unique(np.concatenate([y_train, y_val]))
                self.model.partial_fit(X_train, y_train, classes=unique_classes)
                logger.info("Successfully used partial_fit for incremental training")
                return
            except Exception as e:
                logger.warning(
                    f"partial_fit failed, falling back to standard training: {str(e)}"
                )

        # Special handling for RandomForestClassifier
        if isinstance(self.model, RandomForestClassifier):
            try:
                # Get current parameters
                params = self.model.get_params()

                # Update parameters for the new model
                current_n_estimators = params.get("n_estimators", 100)
                params["n_estimators"] = current_n_estimators + 50  # Add more trees
                params["warm_start"] = True  # Reuse existing trees

                # Create and train a new model with the updated parameters
                new_rf = RandomForestClassifier(**params)

                # Copy any existing estimators if possible
                if hasattr(self.model, "estimators_"):
                    try:
                        new_rf.estimators_ = self.model.estimators_
                    except Exception as e:
                        logger.error(f"Failed to copy existing estimators: {e}")
                        pass

                # Fit the model
                new_rf.fit(X_train, y_train)
                self.model = new_rf
                logger.info(
                    f"Trained RandomForestClassifier with {params['n_estimators']} trees"
                )
                return
            except Exception as e:
                logger.warning(f"RandomForest specific training failed: {str(e)}")

        # Standard training for other models
        try:
            self.model.fit(X_train, y_train)
            logger.info("Model retrained with standard fit method")
        except Exception as e:
            logger.error(f"Standard model training failed: {str(e)}")
            raise


def main():
    """Main entry point for the AI Analyzer service."""
    analyzer = AIAnalyzer()
    analyzer.start()


if __name__ == "__main__":
    main()
