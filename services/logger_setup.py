import logging


def setup_logger(name: str = "echo_pilot", level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with console handler only.

    Args:
        name: Logger name (default: "echo_pilot")
        level: Logging level (default: "INFO")

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(levelname)s - %(asctime)s - %(filename)s:%(lineno)d - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Global logger instance
logger = setup_logger()