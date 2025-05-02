import os
import json
import pika
import uuid
import time
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.logger.logger_settings import api_logger as logger
from config.yaml_config_loader import YamlConfigLoader
from api_gateway.metrics import (
    setup_metrics,
    increment_events_processed,
    set_active_connections,
    SQLI_ATTEMPTS,
    XSS_ATTEMPTS,
    TRAVERSAL_ATTEMPTS,
    DDOS_ATTEMPTS,
    BLOCKED_REQUESTS,
    BLOCKED_IPS,
)
from alert_service.alert_service import router as alert_service_router

# Load configuration
config_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config.yaml",
)
config = YamlConfigLoader(config_path)

# Global settings
SECRET_KEY = config.get(
    "api_gateway.jwt_secret", "change_this_to_secure_key_in_production"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# DDoS protection settings
ERROR_THRESHOLD = config.get("api_gateway.error_threshold", 5)  # Errors before blocking
ERROR_WINDOW = config.get("api_gateway.error_window", 60)  # Time window in seconds
BLOCK_DURATION = config.get(
    "api_gateway.block_duration", 300
)  # Block duration in seconds
REQUEST_THRESHOLD = config.get(
    "api_gateway.request_threshold", 100
)  # Max requests per window
REQUEST_WINDOW = config.get("api_gateway.request_window", 10)  # Time window in seconds

ip_error_counters = {}  # {ip: [timestamp1, timestamp2, ...]}
blocked_ips = {}  # {ip: expiry_timestamp}
ip_request_counters = {}  # {ip: [timestamp1, timestamp2, ...]}
sqli_patterns = []  # Loaded patterns for SQLi detection
xss_patterns = []  # Loaded patterns for XSS detection
traversal_patterns = []  # Loaded patterns for Path Traversal


# Helper function to load SQLi patterns
def load_sqli_patterns():
    """Load SQLi patterns from the patterns file."""
    global sqli_patterns
    try:
        patterns_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "patterns",
            "sql_injections.txt",
        )
        if os.path.exists(patterns_path):
            with open(patterns_path, "r") as f:
                sqli_patterns = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
            logger.info(f"Loaded {len(sqli_patterns)} SQLi patterns.")
        else:
            logger.warning(f"SQLi patterns file not found at {patterns_path}")
            sqli_patterns = []
    except Exception as e:
        logger.error(f"Error loading SQLi patterns: {str(e)}")
        sqli_patterns = []


# Helper function to load XSS patterns
def load_xss_patterns():
    """Load XSS patterns from the patterns file."""
    global xss_patterns
    try:
        patterns_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "patterns",
            "xss.txt",  # Point to the XSS patterns file
        )
        if os.path.exists(patterns_path):
            with open(patterns_path, "r") as f:
                xss_patterns = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
            logger.info(f"Loaded {len(xss_patterns)} XSS patterns.")
        else:
            logger.warning(f"XSS patterns file not found at {patterns_path}")
            xss_patterns = []
    except Exception as e:
        logger.error(f"Error loading XSS patterns: {str(e)}")
        xss_patterns = []


# Helper function to load Path Traversal patterns
def load_traversal_patterns():
    """Load Path Traversal patterns from the patterns file."""
    global traversal_patterns
    try:
        patterns_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "patterns",
            "traversal.txt",  # Point to the traversal patterns file
        )
        if os.path.exists(patterns_path):
            with open(patterns_path, "r") as f:
                # Basic patterns like '../', '..\', encoded variants are key
                # More specific file paths might be less effective as generic patterns
                # but are included for completeness
                traversal_patterns = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
            logger.info(f"Loaded {len(traversal_patterns)} Path Traversal patterns.")
        else:
            logger.warning(f"Path Traversal patterns file not found at {patterns_path}")
            traversal_patterns = []
    except Exception as e:
        logger.error(f"Error loading Path Traversal patterns: {str(e)}")
        traversal_patterns = []


# Helper function to check for SQLi
def is_sqli(payload: Dict[str, Any]) -> bool:
    """Check if the payload contains known SQLi patterns."""
    if not isinstance(payload, dict) or not sqli_patterns:
        return False

    for value in payload.values():
        if isinstance(value, str):
            # More robust check: iterate through loaded patterns
            for pattern in sqli_patterns:
                # Basic substring check, can be enhanced with regex if needed
                if pattern in value:
                    logger.warning(
                        f"SQLi pattern detected: '{pattern}' in value '{value[:50]}...'"
                    )
                    return True
        elif isinstance(value, dict):
            # Recursively check nested dictionaries
            if is_sqli(value):
                return True
        elif isinstance(value, list):
            # Check strings within lists
            for item in value:
                if isinstance(item, str):
                    for pattern in sqli_patterns:
                        if pattern in item:
                            logger.warning(
                                f"SQLi pattern detected: '{pattern}' in list item '{item[:50]}...'"
                            )
                            return True
                elif isinstance(item, dict):
                    # Recursively check dictionaries within lists
                    if is_sqli(item):
                        return True

    return False


# Helper function to check for XSS
def is_xss(payload: Dict[str, Any]) -> bool:
    """Check if the payload contains known XSS patterns."""
    if not isinstance(payload, dict) or not xss_patterns:
        return False

    for value in payload.values():
        if isinstance(value, str):
            # Basic substring check for XSS patterns
            for pattern in xss_patterns:
                # Case-insensitive check might be useful for XSS
                if pattern.lower() in value.lower():
                    logger.warning(
                        f"XSS pattern detected: '{pattern}' in value '{value[:50]}...'"
                    )
                    return True
        elif isinstance(value, dict):
            # Recursively check nested dictionaries
            if is_xss(value):
                return True
        elif isinstance(value, list):
            # Check strings within lists
            for item in value:
                if isinstance(item, str):
                    for pattern in xss_patterns:
                        if pattern.lower() in item.lower():
                            logger.warning(
                                f"XSS pattern detected: '{pattern}' in list item '{item[:50]}...'"
                            )
                            return True
                elif isinstance(item, dict):
                    # Recursively check dictionaries within lists
                    if is_xss(item):
                        return True
    return False


# Helper function to check for Path Traversal
def is_path_traversal(payload: Dict[str, Any]) -> bool:
    """Check if the payload contains known Path Traversal patterns."""
    if not isinstance(payload, dict) or not traversal_patterns:
        return False

    # Define keys that might typically contain file paths or resource IDs
    path_related_keys = [
        "file",
        "path",
        "page",
        "include",
        "document",
        "folder",
        "root",
        "dir",
        "template",
        "resource",
        "url",
    ]

    for key, value in payload.items():
        # Check only in specific keys or if the value itself looks like a path
        is_path_key = any(p_key in key.lower() for p_key in path_related_keys)

        if isinstance(value, str):
            # Prioritize checking path-related keys
            if is_path_key:
                # Check for common traversal sequences
                for pattern in traversal_patterns:
                    # URL Decode might be needed here in a real-world scenario
                    # For simplicity, we'll do a direct substring check
                    if pattern in value:
                        logger.warning(
                            f"Path Traversal pattern detected: '{pattern}' in key '{key}', value '{value[:50]}...'"
                        )
                        return True
        elif isinstance(value, dict):
            # Recursively check nested dictionaries
            if is_path_traversal(value):
                return True
        # We might not need to check lists as deeply for path traversal
        # unless list items are expected to be paths themselves.

    return False


# Create FastAPI app
app = FastAPI(
    title="Threat Analysis API Gateway",
    description="API Gateway for the Threat Analysis Service",
    version="1.0.0",
)
app.include_router(alert_service_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api_gateway.cors.allowed_origins", ["*"]),
    allow_credentials=True,
    allow_methods=config.get("api_gateway.cors.allowed_methods", ["*"]),
    allow_headers=["*"],
)

# Load patterns on startup
load_sqli_patterns()
load_xss_patterns()
load_traversal_patterns()  # Load Path Traversal patterns

# Setup metrics
setup_metrics(app)


# Add IP blocking middleware
@app.middleware("http")
async def block_banned_ips(request: Request, call_next):
    """Middleware to block banned IPs and DDoS attacks"""
    client_host = request.client.host if request.client else "unknown"

    # Check if IP is blocked
    if is_ip_blocked(client_host):
        logger.warning(f"Blocked request from banned IP: {client_host}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "Your IP address has been temporarily blocked due to suspicious activity."
            },
        )

    # Track request rate
    current_time = time.time()
    if client_host not in ip_request_counters:
        ip_request_counters[client_host] = []

    # Remove old requests outside the window
    ip_request_counters[client_host] = [
        t for t in ip_request_counters[client_host] if current_time - t < REQUEST_WINDOW
    ]

    # Add current request
    ip_request_counters[client_host].append(current_time)

    # Block if threshold exceeded
    if len(ip_request_counters[client_host]) > REQUEST_THRESHOLD:
        expiry_time = current_time + BLOCK_DURATION
        blocked_ips[client_host] = expiry_time
        DDOS_ATTEMPTS.labels(source_ip=client_host, attack_type="rate_limit").inc()
        BLOCKED_REQUESTS.labels(block_type="ddos").inc()
        BLOCKED_IPS.labels(reason="ddos").inc()
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Request rate limit exceeded."},
        )

    # Process the request normally
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in middleware: {error_msg}")
        if track_error(client_host, error_msg):
            logger.warning(f"Blocked IP {client_host} due to repeated errors")
        raise


# OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Pydantic models for request validation
class ThreatEvent(BaseModel):
    source_ip: str = Field(..., description="Source IP address of the event")
    destination_ip: Optional[str] = Field(None, description="Destination IP address")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Event timestamp"
    )
    event_type: str = Field(
        ..., description="Type of event (e.g., 'connection', 'auth')"
    )
    payload: Dict[str, Any] = Field(..., description="Event payload with details")
    source_type: str = Field(
        ..., description="Source of the event (e.g., 'IDS', 'Firewall')"
    )


class AnalysisResult(BaseModel):
    event_id: str
    is_threat: bool
    confidence: float
    threat_type: Optional[str] = None
    timestamp: datetime
    details: Dict[str, Any] = {}


class User(BaseModel):
    username: str
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


# For demo purposes - in production, use a proper user store
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Simple password verification function."""
    # In production, use proper password hashing with bcrypt
    return (
        plain_password == "secret"
        and hashed_password
        == "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
    )


def get_user(db: Dict[str, Dict[str, Any]], username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username not in db:
        return None
    user_dict = db[username]
    return UserInDB(**user_dict)


def authenticate_user(
    fake_db: Dict[str, Dict[str, Any]], username: str, password: str
) -> Union[UserInDB, bool]:
    """Authenticate a user."""
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")  # type: ignore
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    if token_data.username is None:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# RabbitMQ connection management
def get_rabbitmq_connection():
    """Create and return a connection to RabbitMQ"""
    rabbitmq_host = config.get("rabbitmq.host", "localhost")
    rabbitmq_port = config.get("rabbitmq.port", 5672)
    rabbitmq_user = config.get("rabbitmq.user", "guest")
    rabbitmq_pass = config.get("rabbitmq.password", "guest")

    try:
        credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_pass)
        parameters = pika.ConnectionParameters(
            host=rabbitmq_host,
            port=rabbitmq_port,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
        )
        connection = pika.BlockingConnection(parameters)
        set_active_connections(1)  # Set active connections metric
        logger.info(f"Connected to RabbitMQ at {rabbitmq_host}:{rabbitmq_port}")
        return connection
    except Exception as e:
        set_active_connections(0)  # Set active connections metric
        logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
        raise ConnectionError(f"Failed to connect to message queue: {str(e)}")


def publish_to_rabbitmq(queue_name: str, message: Dict[str, Any]) -> str:
    """Publish a message to RabbitMQ queue"""
    try:
        connection = get_rabbitmq_connection()
        channel = connection.channel()

        # Declare the queue (creates if doesn't exist)
        channel.queue_declare(queue=queue_name, durable=True)

        # Generate a unique ID for the message
        message_id = str(uuid.uuid4())
        if "id" not in message:
            message["id"] = message_id

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        # Publish message with custom JSON serializer for datetime objects
        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=json.dumps(message, default=json_serializer).encode(),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                message_id=message_id,
                content_type="application/json",
            ),
        )

        logger.info(f"Published message {message_id} to queue {queue_name}")
        increment_events_processed("success")  # Increment success metric
        connection.close()
        return message_id
    except Exception as e:
        logger.error(f"Error publishing to RabbitMQ: {str(e)}")
        increment_events_processed("error")  # Increment error metric
        raise


@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Dict[str, str]:
    """Login endpoint to get JWT token."""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user or not isinstance(user, UserInDB):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {"message": "Threat Analysis Service API Gateway"}


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    # Check RabbitMQ connection
    try:
        connection = get_rabbitmq_connection()
        connection.close()
        rabbitmq_status = "healthy"
    except Exception as e:
        rabbitmq_status = f"unhealthy: {str(e)}"

    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "services": {"api_gateway": "healthy", "rabbitmq": rabbitmq_status},
    }


@app.post(
    "/events", response_model=Dict[str, Any], status_code=status.HTTP_202_ACCEPTED
)
async def receive_event(
    event: ThreatEvent,
    current_user: User = Depends(get_current_active_user),
    request: Request = None,
) -> Dict[str, Any]:
    """
    Receive a security event for threat analysis.

    This endpoint accepts security events from various sources, performs an immediate
    SQLi check, blocks if necessary, and queues valid events for analysis.
    """
    client_ip = event.source_ip

    # 1. Check if the IP is already blocked (from previous errors or DDoS)
    if is_ip_blocked(client_ip):
        logger.warning(f"Rejected request from already blocked IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Your IP address has been temporarily blocked due to suspicious activity. Please try again later.",
        )

    # 2. Perform immediate SQLi check on the payload
    if is_sqli(event.payload):
        detected_pattern = next(
            (p for p in sqli_patterns if p in str(event.payload)), "unknown"
        )
        SQLI_ATTEMPTS.labels(source_ip=client_ip, pattern=detected_pattern).inc()
        BLOCKED_REQUESTS.labels(block_type="sqli").inc()
        BLOCKED_IPS.labels(reason="sqli").inc()
        logger.warning(f"SQLi attempt detected from IP: {client_ip}. Blocking IP.")
        # Immediately block the IP for SQLi attempt
        expiry_time = time.time() + BLOCK_DURATION
        blocked_ips[client_ip] = expiry_time
        # Optionally, increment a specific metric here
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request blocked due to potential security threat.",
        )

    # 3. Perform immediate XSS check on the payload
    if is_xss(event.payload):
        detected_pattern = next(
            (p for p in xss_patterns if p in str(event.payload)), "unknown"
        )
        XSS_ATTEMPTS.labels(source_ip=client_ip, pattern=detected_pattern).inc()
        BLOCKED_REQUESTS.labels(block_type="xss").inc()
        BLOCKED_IPS.labels(reason="xss").inc()
        logger.warning(f"XSS attempt detected from IP: {client_ip}. Blocking IP.")
        # Immediately block the IP for XSS attempt
        expiry_time = time.time() + BLOCK_DURATION
        blocked_ips[client_ip] = expiry_time
        # Optionally, increment an XSS metric here
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request blocked due to potential security threat (XSS).",
        )

    # 4. Perform immediate Path Traversal check on the payload
    if is_path_traversal(event.payload):
        detected_pattern = next(
            (p for p in traversal_patterns if p in str(event.payload)), "unknown"
        )
        TRAVERSAL_ATTEMPTS.labels(source_ip=client_ip, pattern=detected_pattern).inc()
        BLOCKED_REQUESTS.labels(block_type="path_traversal").inc()
        BLOCKED_IPS.labels(reason="path_traversal").inc()
        expiry_time = time.time() + BLOCK_DURATION
        blocked_ips[client_ip] = expiry_time
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request blocked due to potential security threat (Path Traversal).",
        )

    # 5. If not blocked and no immediate threat, proceed to queue
    try:
        event_dict = event.dict()
        event_id = publish_to_rabbitmq("threat_events", event_dict)
        logger.info(f"Received event from {event.source_type}, queued as {event_id}")
        return {
            "status": "accepted",
            "event_id": event_id,
            "message": "Event accepted for analysis",
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing event: {error_msg}")

        # Track the error and block the IP if needed
        if track_error(client_ip, error_msg):
            logger.warning(f"Blocked IP {client_ip} due to repeated errors")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process event: {error_msg}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler to log and return proper error responses"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred"},
    )


# JSON serializer for datetime objects
def json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} is not JSON serializable")


# Helper functions for error tracking and IP blocking
def is_ip_blocked(ip_address: str) -> bool:
    """
    Check if an IP is currently blocked.

    Args:
        ip_address: IP address to check

    Returns:
        True if the IP is blocked, False otherwise
    """
    # Clean up expired blocks
    current_time = time.time()
    expired_ips = [ip for ip, expiry in blocked_ips.items() if expiry <= current_time]
    for ip in expired_ips:
        del blocked_ips[ip]
        logger.info(f"Unblocked IP: {ip} (block expired)")

    # Check if IP is blocked
    if ip_address in blocked_ips:
        if current_time < blocked_ips[ip_address]:
            return True

    return False


def track_error(ip_address: str, error_message: str) -> bool:
    """
    Track errors from an IP and block if threshold is exceeded.

    Args:
        ip_address: Source IP address
        error_message: Error message

    Returns:
        True if the IP was blocked, False otherwise
    """
    current_time = time.time()

    if ip_address not in ip_error_counters:
        ip_error_counters[ip_address] = []

    ip_error_counters[ip_address] = [
        t for t in ip_error_counters[ip_address] if current_time - t < ERROR_WINDOW
    ]

    ip_error_counters[ip_address].append(current_time)

    if len(ip_error_counters[ip_address]) >= ERROR_THRESHOLD:
        expiry_time = current_time + BLOCK_DURATION
        blocked_ips[ip_address] = expiry_time
        ip_error_counters[ip_address] = []
        logger.warning(f"BLOCKED IP: {ip_address} for {BLOCK_DURATION} sec")
        return True
    return False


if __name__ == "__main__":
    import uvicorn

    port = config.get("api_gateway.port", 8000)

    logger.info(f"Starting API Gateway on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
