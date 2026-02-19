"""Entry point for the credit risk scoring API server."""

import uvicorn

from src.api.app import create_app
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Start the FastAPI server with configuration from config.yaml."""
    config = load_config()
    app = create_app(config)
    logger.info(
        "Starting Credit Risk Scoring API on %s:%s",
        config.api["host"],
        config.api["port"],
    )
    uvicorn.run(app, host=config.api["host"], port=config.api["port"])


if __name__ == "__main__":
    main()
