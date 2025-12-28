import sys
from loguru import logger


def setup_logger() -> None:
    logger.remove()

    logger.add(
        sink=sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
