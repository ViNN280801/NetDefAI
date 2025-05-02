import os
import json
from datetime import datetime
from src.logger.logger_settings import db_logger as logger
from src.config.yaml_config_loader import YamlConfigLoader
from typing import Dict, Any, List, Optional, Union, Tuple

# Load configuration
config_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config.yaml",
)
config = YamlConfigLoader(config_path)


class CassandraConnector:
    """
    Connector class for Apache Cassandra database.
    Handles connections, data insertion and retrieval.
    """

    def __init__(self):
        """Initialize the Cassandra connector with configuration."""
        self.hosts = config.get("cassandra.hosts", ["localhost"])
        self.port = config.get("cassandra.port", 9042)
        self.keyspace = config.get("cassandra.keyspace", "threat_analysis")
        self.username = config.get("cassandra.username", None)
        self.password = config.get("cassandra.password", None)
        self.replication_factor = config.get("cassandra.replication_factor", 3)

        self.cluster = None
        self.session = None

        # Connect to Cassandra
        self._connect()

        # Initialize schema
        self._init_schema()

        logger.info(f"Cassandra connector initialized with keyspace {self.keyspace}")

    def _connect(self):
        """Connect to the Cassandra cluster."""
        try:
            # Import is here to avoid dependency error if not needed
            from cassandra.cluster import (
                Cluster,
                ExecutionProfile,
                EXEC_PROFILE_DEFAULT,
            )
            from cassandra.auth import PlainTextAuthProvider
            from cassandra import ConsistencyLevel

            # Create auth provider if credentials are provided
            auth_provider = None
            if self.username and self.password:
                auth_provider = PlainTextAuthProvider(
                    username=self.username, password=self.password
                )

            # Configure execution profile
            execution_profile = ExecutionProfile(
                consistency_level=ConsistencyLevel.LOCAL_QUORUM,
                request_timeout=15.0,
                row_factory=dict_factory,
            )

            # Connect to cluster
            self.cluster = Cluster(
                contact_points=self.hosts,
                port=self.port,
                auth_provider=auth_provider,
                execution_profiles={EXEC_PROFILE_DEFAULT: execution_profile},
            )

            self.session = self.cluster.connect()
            logger.info(f"Connected to Cassandra cluster at {self.hosts}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Cassandra: {str(e)}")
            raise

    def _init_schema(self):
        """Initialize the keyspace and tables if they don't exist."""
        try:
            if not self.session:
                msg = "Cassandra session not initialized"
                logger.error(msg)
                raise Exception(msg)

            # Create keyspace if not exists
            self.session.execute(
                f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': {self.replication_factor}}}
            """
            )

            # Use keyspace
            self.session.execute(f"USE {self.keyspace}")

            # Create incidents table for main threat events
            self.session.execute(
                """
            CREATE TABLE IF NOT EXISTS incidents (
                event_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                source_ip TEXT,
                is_threat BOOLEAN,
                confidence FLOAT,
                threat_type TEXT,
                details TEXT
            )
            """
            )

            # Create events_by_ip table for querying by IP address
            self.session.execute(
                """
            CREATE TABLE IF NOT EXISTS events_by_ip (
                source_ip TEXT,
                timestamp TIMESTAMP,
                event_id TEXT,
                is_threat BOOLEAN,
                confidence FLOAT,
                threat_type TEXT,
                PRIMARY KEY ((source_ip), timestamp, event_id)
            ) WITH CLUSTERING ORDER BY (timestamp DESC)
            """
            )

            # Create events_by_day table for time-based analytics
            self.session.execute(
                """
            CREATE TABLE IF NOT EXISTS events_by_day (
                day TEXT,
                timestamp TIMESTAMP,
                event_id TEXT,
                source_ip TEXT,
                is_threat BOOLEAN,
                threat_type TEXT,
                PRIMARY KEY ((day), timestamp, event_id)
            ) WITH CLUSTERING ORDER BY (timestamp DESC)
            """
            )

            logger.info(f"Schema initialized for keyspace {self.keyspace}")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {str(e)}")
            raise

    def insert_event(
        self,
        event_id: str,
        timestamp: str,
        source_ip: str,
        is_threat: bool,
        confidence: float,
        threat_type: str,
        details: str,
    ):
        """
        Insert an event into Cassandra tables.

        Args:
            event_id: Unique identifier for the event
            timestamp: ISO formatted timestamp
            source_ip: Source IP address
            is_threat: Whether the event is a threat
            confidence: Confidence score (0-1)
            threat_type: Type of threat
            details: JSON string with full details
        """
        try:
            # Convert ISO timestamp to datetime
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                dt = timestamp

            # Extract day for time-based partitioning
            day = dt.strftime("%Y-%m-%d")

            if not self.session:
                msg = "Cassandra session not initialized"
                logger.error(msg)
                raise Exception(msg)

            # Insert into incidents table
            self.session.execute(
                """
                INSERT INTO incidents (
                    event_id, timestamp, source_ip, is_threat, confidence, threat_type, details
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (event_id, dt, source_ip, is_threat, confidence, threat_type, details),
            )

            # Insert into events_by_ip table
            self.session.execute(
                """
                INSERT INTO events_by_ip (
                    source_ip, timestamp, event_id, is_threat, confidence, threat_type
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (source_ip, dt, event_id, is_threat, confidence, threat_type),
            )

            # Insert into events_by_day table
            self.session.execute(
                """
                INSERT INTO events_by_day (
                    day, timestamp, event_id, source_ip, is_threat, threat_type
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (day, dt, event_id, source_ip, is_threat, threat_type),
            )

            logger.info(f"Event {event_id} inserted into Cassandra")
        except Exception as e:
            logger.error(f"Failed to insert event {event_id}: {str(e)}")
            raise

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an event by its ID.

        Args:
            event_id: The event ID to retrieve

        Returns:
            Dictionary with event data or None if not found
        """
        try:
            if not self.session:
                msg = "Cassandra session not initialized"
                logger.error(msg)
                raise Exception(msg)

            rows = self.session.execute(
                "SELECT * FROM incidents WHERE event_id = %s", (event_id,)
            )

            for row in rows:
                # Parse details JSON
                if "details" in row and row["details"]:
                    try:
                        row["details"] = json.loads(row["details"])
                    except json.JSONDecodeError:
                        pass

                return row

            return None
        except Exception as e:
            logger.error(f"Failed to get event {event_id}: {str(e)}")
            raise

    def get_events_by_ip(
        self, source_ip: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve events by source IP.

        Args:
            source_ip: The source IP to search for
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        try:
            if not self.session:
                msg = "Cassandra session not initialized"
                logger.error(msg)
                raise Exception(msg)

            rows = self.session.execute(
                "SELECT * FROM events_by_ip WHERE source_ip = %s LIMIT %s",
                (source_ip, limit),
            )

            return list(rows)
        except Exception as e:
            logger.error(f"Failed to get events for IP {source_ip}: {str(e)}")
            raise

    def get_events_by_day(self, day: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve events by day.

        Args:
            day: Day in format YYYY-MM-DD
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        try:
            if not self.session:
                msg = "Cassandra session not initialized"
                logger.error(msg)
                raise Exception(msg)

            rows = self.session.execute(
                "SELECT * FROM events_by_day WHERE day = %s LIMIT %s", (day, limit)
            )

            return list(rows)
        except Exception as e:
            logger.error(f"Failed to get events for day {day}: {str(e)}")
            raise

    def get_recent_threats(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recent threat events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of threat event dictionaries
        """
        try:
            if not self.session:
                msg = "Cassandra session not initialized"
                logger.error(msg)
                raise Exception(msg)

            rows = self.session.execute(
                "SELECT * FROM incidents WHERE is_threat = True LIMIT %s ALLOW FILTERING",
                (limit,),
            )

            return list(rows)
        except Exception as e:
            logger.error(f"Failed to get recent threats: {str(e)}")
            raise

    def close(self):
        """Close the Cassandra connection."""
        if self.cluster:
            try:
                self.cluster.shutdown()
                logger.info("Cassandra connection closed")
            except Exception as e:
                logger.error(f"Error closing Cassandra connection: {str(e)}")


class RedisConnector:
    """
    Connector class for Redis cache.
    Handles connections, caching and retrieval.
    """

    def __init__(self):
        """Initialize the Redis connector with configuration."""
        self.host = config.get("redis.host", "localhost")
        self.port = config.get("redis.port", 6379)
        self.db = config.get("redis.db", 0)
        self.password = config.get("redis.password", None)
        self.client = None

        # Connect to Redis
        self._connect()

        logger.info(f"Redis connector initialized at {self.host}:{self.port}")

    def _connect(self):
        """Connect to Redis."""
        try:
            # Import is here to avoid dependency error if not needed
            import redis

            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,  # Auto-decode to str
                socket_timeout=5,
                socket_connect_timeout=5,
            )

            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    def set(self, key: str, value: str, ttl: int) -> bool:
        """
        Set a string value in Redis.

        Args:
            key: Redis key
            value: String value
            ttl: Time-to-live in seconds (optional)

        Returns:
            Success flag
        """
        try:
            if not self.client:
                msg = "Redis client not initialized"
                logger.error(msg)
                raise Exception(msg)

            self.client.set(key, value, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to set key {key} in Redis: {str(e)}")
            return False

    def get(self, key: str) -> Optional[str]:
        """
        Get a string value from Redis.

        Args:
            key: Redis key

        Returns:
            String value or None if not found
        """
        try:
            if not self.client:
                msg = "Redis client not initialized"
                logger.error(msg)
                raise Exception(msg)

            return self.client.get(key)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to get key {key} from Redis: {str(e)}")
            return None

    def set_json(self, key: str, obj: Dict[str, Any], ttl: int) -> bool:
        """
        Store a JSON serializable object in Redis.

        Args:
            key: Redis key
            obj: Object to serialize and store
            ttl: Time-to-live in seconds (optional)

        Returns:
            Success flag
        """
        try:
            json_data = json.dumps(obj)
            return self.set(key, json_data, ttl)
        except Exception as e:
            logger.error(f"Failed to set JSON for key {key} in Redis: {str(e)}")
            return False

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and deserialize a JSON object from Redis.

        Args:
            key: Redis key

        Returns:
            Deserialized object or None if not found
        """
        try:
            data = self.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get JSON for key {key} from Redis: {str(e)}")
            return None

    def add_to_set(self, key: str, member: str) -> bool:
        """
        Add a member to a Redis set.

        Args:
            key: Redis key for the set
            member: Member to add

        Returns:
            Success flag
        """
        try:
            if not self.client:
                msg = "Redis client not initialized"
                logger.error(msg)
                raise Exception(msg)

            self.client.sadd(key, member)
            return True
        except Exception as e:
            logger.error(f"Failed to add {member} to set {key}: {str(e)}")
            return False

    def get_set_members(self, key: str) -> List[str]:
        """
        Get all members of a Redis set.

        Args:
            key: Redis key for the set

        Returns:
            List of set members
        """
        try:
            if not self.client:
                msg = "Redis client not initialized"
                logger.error(msg)
                raise Exception(msg)

            return list(self.client.smembers(key))  # type: ignore
        except Exception as e:
            logger.error(f"Failed to get members of set {key}: {str(e)}")
            return []

    def add_to_sorted_set(
        self, key: str, member: str, score: float, max_size: int
    ) -> bool:
        """
        Add a member to a sorted set with score.

        Args:
            key: Redis key for the sorted set
            member: Member to add
            score: Score for ordering
            max_size: Maximum size of the set (prunes oldest if exceeded)

        Returns:
            Success flag
        """
        try:
            if not self.client:
                msg = "Redis client not initialized"
                logger.error(msg)
                raise Exception(msg)

            # Add to sorted set
            self.client.zadd(key, {member: score})

            # Trim to max size if specified
            if max_size and max_size > 0:
                current_size = self.client.zcard(key)
                if current_size > max_size:  # type: ignore
                    # Remove oldest entries (lowest score)
                    excess = current_size - max_size  # type: ignore
                    self.client.zremrangebyrank(key, 0, excess - 1)

            return True
        except Exception as e:
            logger.error(f"Failed to add {member} to sorted set {key}: {str(e)}")
            return False

    def get_sorted_set_members(
        self, key: str, start: int = 0, end: int = -1, with_scores: bool = False
    ) -> List[Union[str, Tuple[str, float]]]:
        """
        Get members from a sorted set.

        Args:
            key: Redis key for the sorted set
            start: Start index
            end: End index (-1 for all)
            with_scores: Whether to include scores

        Returns:
            List of members or (member, score) tuples
        """
        try:
            if not self.client:
                msg = "Redis client not initialized"
                logger.error(msg)
                raise Exception(msg)

            if with_scores:
                result = self.client.zrange(key, start, end, withscores=True)
                return result  # type: ignore
            else:
                return self.client.zrange(key, start, end)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to get members of sorted set {key}: {str(e)}")
            return []

    def close(self):
        """Close the Redis connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")


# Helper function for Cassandra to return rows as dictionaries
def dict_factory(colnames, rows):
    """Convert Cassandra row to dictionary."""
    return [dict(zip(colnames, row)) for row in rows]
