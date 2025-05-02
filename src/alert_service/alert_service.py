import os
import json
import time
import pika
import uuid
import signal
import smtplib
import requests
import threading
from typing import Dict, Any
from datetime import datetime
from email.message import EmailMessage
from src.logger import alert_logger as logger
from fastapi import APIRouter, Request, HTTPException
from src.config.yaml_config_loader import YamlConfigLoader


# Load configuration
config_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config.yaml",
)
config = YamlConfigLoader(config_path)

router = APIRouter()


@router.post("/webhook")
async def handle_webhook(request: Request):
    try:
        alert_data = await request.json()
        logger.info(f"Received alert: {json.dumps(alert_data, indent=2)}")

        for alert in alert_data.get("alerts", []):
            status = alert.get("status", "firing")
            labels = alert.get("labels", {})
            annotations = alert.get("annotations", {})

            logger.warning(
                f"Alert: status={status}, "
                f"name={labels.get('alertname')}, "
                f"severity={labels.get('severity')}, "
                f"summary={annotations.get('summary')}"
            )

            _send_alert_notification(alert)

        return {"status": "success"}

    except json.JSONDecodeError:
        logger.error("Invalid JSON received in webhook")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"Webhook processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


class AlertService:
    """
    Alert Service that processes threat alerts from RabbitMQ queue
    and sends notifications through various channels (Email, Webhook, etc).
    """

    def __init__(self):
        """Initialize the Alert Service with configuration and connections."""
        logger.info("Initializing Alert Service")

        # RabbitMQ configuration
        self.rabbitmq_host = config.get("rabbitmq.host", "localhost")
        self.rabbitmq_port = config.get("rabbitmq.port", 5672)
        self.rabbitmq_user = config.get("rabbitmq.user", "guest")
        self.rabbitmq_pass = config.get("rabbitmq.password", "guest")
        self.alerts_queue = config.get("rabbitmq.queues.alerts", "alerts")

        # Email configuration
        self.email_enabled = config.get("alert_service.email.enabled", False)
        self.email_from = config.get("alert_service.email.from", "alerts@example.com")
        self.email_to = config.get("alert_service.email.to", [])
        self.email_server = config.get("alert_service.email.server", "localhost")
        self.email_port = config.get("alert_service.email.port", 25)
        self.email_user = config.get("alert_service.email.user", None)
        self.email_password = config.get("alert_service.email.password", None)
        self.email_use_tls = config.get("alert_service.email.use_tls", False)

        # Webhook configuration
        self.webhook_enabled = config.get("alert_service.webhook.enabled", False)
        self.webhooks = config.get("alert_service.webhook.urls", [])

        # Telegram configuration
        self.telegram_enabled = config.get("alert_service.telegram.enabled", False)
        self.telegram_token = config.get("alert_service.telegram.token", "")
        self.telegram_chat_ids = config.get("alert_service.telegram.chat_ids", [])

        # Slack configuration
        self.slack_enabled = config.get("alert_service.slack.enabled", False)
        self.slack_webhooks = config.get("alert_service.slack.webhooks", [])

        # Thread control
        self.should_exit = threading.Event()
        self.connection = None
        self.channel = None

        # Validation
        if not any(
            [
                self.email_enabled,
                self.webhook_enabled,
                self.telegram_enabled,
                self.slack_enabled,
            ]
        ):
            logger.warning(
                "No notification channels are enabled. Alerts will be processed but not sent."
            )

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

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

            # Declare queue
            self.channel.queue_declare(queue=self.alerts_queue, durable=True)

            # Set QoS prefetch to control load
            self.channel.basic_qos(prefetch_count=1)

            logger.info(
                f"Connected to RabbitMQ at {self.rabbitmq_host}:{self.rabbitmq_port}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            return False

    def _send_email_alert(self, alert: Dict[str, Any]):
        """
        Send an email alert.

        Args:
            alert: The alert information
        """
        if not self.email_enabled or not self.email_to:
            return

        try:
            # Create email message
            msg = EmailMessage()

            # Set content
            event_id = alert.get("event_id", "unknown")
            threat_type = alert.get("threat_type", "unknown")
            source_ip = alert.get("source_ip", "unknown")
            confidence = alert.get("confidence", 0.0)
            timestamp = alert.get("timestamp", datetime.now().isoformat())

            msg["Subject"] = f"THREAT ALERT: {threat_type} from {source_ip}"
            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_to)

            # Email body
            body = f"""
Threat Alert Notification
-------------------------

Event ID: {event_id}
Type: {threat_type}
Source IP: {source_ip}
Confidence: {confidence:.2f}
Timestamp: {timestamp}

This is an automated alert from the Threat Analysis Service.
"""

            msg.set_content(body)

            # Connect to server and send
            if self.email_use_tls:
                server = smtplib.SMTP(self.email_server, self.email_port)
                server.starttls()
            else:
                server = smtplib.SMTP(self.email_server, self.email_port)

            if self.email_user and self.email_password:
                server.login(self.email_user, self.email_password)

            server.send_message(msg)
            server.quit()

            logger.info(
                f"Email alert sent for event {event_id} to {len(self.email_to)} recipients"
            )
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """
        Send an alert to webhooks.

        Args:
            alert: The alert information
        """
        if not self.webhook_enabled or not self.webhooks:
            return

        for webhook_url in self.webhooks:
            try:
                # Send webhook request
                response = requests.post(
                    webhook_url,
                    json=alert,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )

                if response.status_code < 200 or response.status_code >= 300:
                    logger.error(
                        f"Webhook returned error status {response.status_code}: {response.text}"
                    )
                else:
                    logger.info(f"Webhook alert sent to {webhook_url}")
            except Exception as e:
                logger.error(f"Failed to send webhook alert to {webhook_url}: {str(e)}")

    def _send_telegram_alert(self, alert: Dict[str, Any]):
        """
        Send an alert via Telegram.

        Args:
            alert: The alert information
        """
        if (
            not self.telegram_enabled
            or not self.telegram_token
            or not self.telegram_chat_ids
        ):
            return

        try:
            # Format message
            event_id = alert.get("event_id", "unknown")
            threat_type = alert.get("threat_type", "unknown")
            source_ip = alert.get("source_ip", "unknown")
            confidence = alert.get("confidence", 0.0)
            timestamp = alert.get("timestamp", datetime.now().isoformat())

            message = f"""
                    🚨 *THREAT ALERT* 🚨

                    *Type:* {threat_type}
                    *Source IP:* {source_ip}
                    *Confidence:* {confidence:.2f}
                    *Event ID:* {event_id}
                    *Time:* {timestamp}
                    """

            # Send to each chat ID
            for chat_id in self.telegram_chat_ids:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                }

                response = requests.post(url, json=payload, timeout=10)

                if response.status_code != 200:
                    logger.error(
                        f"Telegram API returned error: {response.status_code} - {response.text}"
                    )
                else:
                    logger.info(f"Telegram alert sent to chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {str(e)}")

    def _send_slack_alert(self, alert: Dict[str, Any]):
        """
        Send an alert to Slack.

        Args:
            alert: The alert information
        """
        if not self.slack_enabled or not self.slack_webhooks:
            return

        try:
            # Format message
            event_id = alert.get("event_id", "unknown")
            threat_type = alert.get("threat_type", "unknown")
            source_ip = alert.get("source_ip", "unknown")
            confidence = alert.get("confidence", 0.0)
            timestamp = alert.get("timestamp", datetime.now().isoformat())

            # Create Slack message payload
            payload = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "🚨 THREAT ALERT 🚨"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Type:* {threat_type}"},
                            {"type": "mrkdwn", "text": f"*Source IP:* {source_ip}"},
                            {
                                "type": "mrkdwn",
                                "text": f"*Confidence:* {confidence:.2f}",
                            },
                            {"type": "mrkdwn", "text": f"*Event ID:* {event_id}"},
                        ],
                    },
                    {
                        "type": "context",
                        "elements": [
                            {"type": "plain_text", "text": f"Time: {timestamp}"}
                        ],
                    },
                ]
            }

            # Send to each webhook
            for webhook_url in self.slack_webhooks:
                response = requests.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Slack webhook returned error: {response.status_code} - {response.text}"
                    )
                else:
                    logger.info("Slack alert sent via webhook")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")

    def _process_alert(self, alert: Dict[str, Any]):
        """
        Process and send an alert through all enabled channels.

        Args:
            alert: The alert information
        """
        try:
            event_id = alert.get("event_id", "unknown")
            threat_type = alert.get("threat_type", "unknown")

            logger.info(
                f"Processing alert for event {event_id}, threat type: {threat_type}"
            )

            if threat_type == "sql_injection":
                logger.warning(
                    f"SQLi attempt detected from IP: {alert.get('source_ip')}"
                )

            # Send alerts through all enabled channels
            self._send_email_alert(alert)
            self._send_webhook_alert(alert)
            self._send_telegram_alert(alert)
            self._send_slack_alert(alert)

            logger.info(f"Alert processing completed for event {event_id}")
        except Exception as e:
            logger.error(f"Error processing alert: {str(e)}")

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
            event_id = message.get("event_id", "unknown")
            logger.info(f"Received alert for event {event_id}")

            # Process the alert
            self._process_alert(message)

            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except json.JSONDecodeError:
            logger.error("Received invalid JSON message")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def start(self):
        """Start the Alert Service."""
        logger.info("Starting Alert Service")

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
            logger.info(
                "Alert Service will start in limited mode (no message processing)"
            )
            return

        # Start consuming messages
        if not self.channel:
            msg = "RabbitMQ channel not initialized"
            logger.error(msg)
            logger.info(
                "Alert Service will start in limited mode (no message processing)"
            )
            return

        try:
            self.channel.basic_consume(
                queue=self.alerts_queue, on_message_callback=self._process_message
            )

            logger.info(f"Waiting for alerts on queue {self.alerts_queue}")

            # Start consumption loop
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Error in consumption loop: {str(e)}")
            self.stop()

    def stop(self):
        """Stop the Alert Service."""
        logger.info("Stopping Alert Service")

        # Signal threads to exit
        self.should_exit.set()

        # Stop consuming
        if self.channel:
            try:
                self.channel.stop_consuming()
            except Exception:
                pass

        # Close connection
        if self.connection and self.connection.is_open:
            try:
                self.connection.close()
            except Exception:
                pass

        logger.info("Alert Service stopped")


def _send_alert_notification(alert: Dict[str, Any]):
    labels = alert.get("labels", {})
    annotations = alert.get("annotations", {})

    message = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "threat_type": labels.get("alertname", "unknown"),
        "severity": labels.get("severity", "medium"),
        "source_ip": labels.get("source_ip", "unknown"),
        "summary": annotations.get("summary", "No details"),
        "description": annotations.get("description", ""),
        "status": alert.get("status", "firing"),
    }
    logger.info(f"Sending alert notification: {message}")

    # Examples (not working in my implementation)
    # _send_email_alert(message)
    # _send_slack_alert(message)
    # _send_telegram_alert(message)


def main():
    """Main entry point for the Alert Service."""
    service = AlertService()
    service.start()


if __name__ == "__main__":
    main()
