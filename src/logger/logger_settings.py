import os
from src.logger import Logger

# Create logs directory if it doesn't exist
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

# Configure root logger
root_logger = Logger(log_level="INFO", caller_module="root", log_file=log_file)

# Dataset generation loggers
path_traversal_dataset_logger = Logger(
    log_level="DEBUG", caller_module="attacks.path_traversal.dataset", log_file=log_file
)

sql_injections_dataset_logger = Logger(
    log_level="DEBUG", caller_module="attacks.sql_injections.dataset", log_file=log_file
)

xss_dataset_logger = Logger(
    log_level="DEBUG", caller_module="attacks.xss.dataset", log_file=log_file
)

dos_dataset_logger = Logger(
    log_level="DEBUG", caller_module="attacks.dos.dataset", log_file=log_file
)

dataset_generator_logger = Logger(
    log_level="DEBUG",
    caller_module="dataset_generator",
    log_file=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), log_dir, log_file
    ),
)

# Model training loggers
path_traversal_model_logger = Logger(
    log_level="INFO", caller_module="attacks.path_traversal.model", log_file=log_file
)

sql_injections_model_logger = Logger(
    log_level="INFO", caller_module="attacks.sql_injections.model", log_file=log_file
)

xss_model_logger = Logger(
    log_level="INFO", caller_module="attacks.xss.model", log_file=log_file
)

dos_model_logger = Logger(
    log_level="INFO", caller_module="attacks.dos.model", log_file=log_file
)

# Universal trainer logger
universal_trainer_logger = Logger(
    log_level="INFO",
    caller_module="model_training.universal_trainer",
    log_file=log_file,
)

# Device manager logger
device_manager_logger = Logger(
    log_level="INFO",
    caller_module="model_training.common.device_manager",
    log_file=log_file,
)

# API loggers
api_logger = Logger(log_level="INFO", caller_module="api", log_file=log_file)

# Security logger (for tracking potential attacks on the system itself)
security_logger = Logger(
    log_level="WARNING", caller_module="security", log_file=log_file
)

# Performance logger (for tracking performance metrics)
performance_logger = Logger(
    log_level="INFO", caller_module="performance", log_file=log_file
)

config_loader_logger = Logger(
    log_level="INFO", caller_module="config_loader", log_file=log_file
)

# Analyzer logger
analyzer_logger = Logger(
    log_level="INFO", caller_module="ai_analyzer", log_file=log_file
)

# Alert logger
alert_logger = Logger(
    log_level="INFO", caller_module="alert_service", log_file=log_file
)

# DB logger
db_logger = Logger(
    log_level="INFO", caller_module="utils.db_connectors", log_file=log_file
)


# Function to get appropriate logger based on attack type
def get_dataset_logger(attack_type):
    """
    Returns the appropriate logger for the given attack type.

    Args:
        attack_type (str): The type of attack ('path_traversal', 'sql_injection', 'xss', 'dos')

    Returns:
        Logger: The appropriate logger instance
    """
    attack_type = attack_type.lower()
    if attack_type == "path_traversal":
        return path_traversal_dataset_logger
    elif attack_type in ("sql_injection", "sql_injections", "sqli"):
        return sql_injections_dataset_logger
    elif attack_type in ("xss", "cross_site_scripting"):
        return xss_dataset_logger
    elif attack_type in ("dos", "denial_of_service"):
        return dos_dataset_logger
    else:
        return root_logger


def get_model_logger(attack_type):
    """
    Returns the appropriate model logger for the given attack type.

    Args:
        attack_type (str): The type of attack ('path_traversal', 'sql_injection', 'xss', 'dos')

    Returns:
        Logger: The appropriate logger instance
    """
    attack_type = attack_type.lower()
    if attack_type == "path_traversal":
        return path_traversal_model_logger
    elif attack_type in ("sql_injection", "sql_injections", "sqli"):
        return sql_injections_model_logger
    elif attack_type in ("xss", "cross_site_scripting"):
        return xss_model_logger
    elif attack_type in ("dos", "denial_of_service"):
        return dos_model_logger
    else:
        return universal_trainer_logger
