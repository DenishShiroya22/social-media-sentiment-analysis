"""Connector interface and application-facing connector errors."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import SocialPost


class ConnectorError(RuntimeError):
    """Base error for safe connector failures."""


class ConnectorConfigurationError(ConnectorError):
    """Connector credentials or settings are missing or invalid."""


class ConnectorAuthenticationError(ConnectorError):
    """Remote authentication failed."""


class ConnectorRateLimitError(ConnectorError):
    """Remote service rejected the request because of rate limits."""


class ConnectorRequestError(ConnectorError):
    """Remote request or response processing failed."""


class SocialConnector(ABC):
    """Small interface shared by future social-data connectors."""

    source: str

    @abstractmethod
    def search(self, query: str, limit: int = 50, **kwargs) -> list[SocialPost]:
        """Return bounded, normalized public posts for a query."""
