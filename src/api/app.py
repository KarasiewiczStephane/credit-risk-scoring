"""FastAPI application factory for the credit risk scoring API."""

from fastapi import FastAPI

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Application configuration. If None, defaults are used.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Credit Risk Scoring API",
        description="Interpretable ML model for loan approval decisions",
        version="1.0.0",
    )

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    return app
