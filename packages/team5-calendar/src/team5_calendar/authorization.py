"""Per-service OAuth state; token refresh never writes into installed files."""

from contextvars import ContextVar
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Protocol

from google.auth.exceptions import GoogleAuthError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from .event_provider import CalendarAuthenticationError

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


class Authorization(Protocol):
    def load(self) -> Credentials: ...


class TokenAuthorization:
    def __init__(self, token_file: str | Path) -> None:
        self.path = Path(token_file).expanduser().resolve()
        self._credentials: Credentials | None = None
        self._lock = Lock()

    def load(self) -> Credentials:
        with self._lock:
            if self._credentials is None:
                try:
                    credentials = Credentials.from_authorized_user_file(str(self.path))
                    if not credentials.has_scopes(SCOPES):
                        raise ValueError
                except OSError, ValueError, AttributeError, TypeError, GoogleAuthError:
                    raise CalendarAuthenticationError from None
                self._credentials = credentials
            credentials = self._credentials
            if not credentials.valid:
                if not credentials.expired or not credentials.refresh_token:
                    raise CalendarAuthenticationError
                _refresh(credentials)
            return credentials


class CredentialsAuthorization:
    """Use the consumer's credentials; refresh updates that same object."""

    def __init__(self, credentials: Credentials) -> None:
        self.credentials = credentials
        self._lock = Lock()

    def load(self) -> Credentials:
        with self._lock:
            credentials = self.credentials
            if not credentials.has_scopes(SCOPES):
                raise CalendarAuthenticationError
            if not credentials.valid:
                # A stored grant may hold only a refresh token; refresh fetches access.
                if not credentials.refresh_token:
                    raise CalendarAuthenticationError
                _refresh(credentials)
            return credentials


def _refresh(credentials: Credentials) -> None:
    try:
        credentials.refresh(partial(Request(), timeout=10))
    except TransportError:
        raise
    except GoogleAuthError:
        raise CalendarAuthenticationError(provider=True) from None
    if not credentials.valid or not credentials.has_scopes(SCOPES):
        raise CalendarAuthenticationError(provider=True)


active_authorization: ContextVar[Authorization | None] = ContextVar(
    "team5_calendar_authorization", default=None
)
