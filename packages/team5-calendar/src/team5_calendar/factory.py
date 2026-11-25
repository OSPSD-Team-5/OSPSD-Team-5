"""Create the existing Google implementation without a web framework."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol

from calendar_contract import (
    CalendarError,
    CalendarService,
    CreateEvent,
    ErrorCode,
    Event,
    ReplaceEvent,
)
from google.oauth2.credentials import Credentials

from .authorization import (
    Authorization,
    CredentialsAuthorization,
    TokenAuthorization,
    active_authorization,
)
from .calendar_service import GoogleCalendarService
from .google_create_events import GoogleEventCreator
from .google_delete_events import GoogleEventDeleter
from .google_events import GoogleEventReader
from .google_update_events import GoogleEventUpdater

_TOKEN_FILE_SETUP = (
    "Run python -m team5_calendar.authorize with your OAuth client and token file."
)
_CREDENTIALS_SETUP = (
    "Authorize the account again for credentials with calendar.events access."
)


class UpcomingCalendarService(CalendarService, Protocol):
    """The shared interface, plus a read of upcoming events."""

    def list_upcoming(self, start: datetime, end: datetime) -> list[Event]:
        """Return timed events that overlap start to end, earliest first.

        start and end must be timezone-aware, with start before end. Recurring
        events appear as their single instances; all-day events are skipped.
        """
        ...


class _ConfiguredCalendarService(GoogleCalendarService):
    def __init__(self, authorization: Authorization, setup: str) -> None:
        reader = GoogleEventReader()
        super().__init__(
            creator=GoogleEventCreator(),
            reader=reader,
            updater=GoogleEventUpdater(),
            deleter=GoogleEventDeleter(),
        )
        self._events = reader
        self._authorization = authorization
        self._setup = setup

    @contextmanager
    def _configured(self) -> Iterator[None]:
        token = active_authorization.set(self._authorization)
        try:
            yield
        except CalendarError as error:
            if error.code == ErrorCode.PROVIDER_UNAVAILABLE and not error.retryable:
                raise CalendarError(
                    error.code,
                    f"Calendar authorization is unavailable. {self._setup}",
                    retryable=False,
                    details=error.details,
                ) from None
            raise
        finally:
            active_authorization.reset(token)

    def create(self, request: CreateEvent) -> Event:
        with self._configured():
            return super().create(request)

    def get(self, event_id: str) -> Event:
        with self._configured():
            return super().get(event_id)

    def replace(self, event_id: str, request: ReplaceEvent) -> Event:
        with self._configured():
            return super().replace(event_id, request)

    def delete(self, event_id: str) -> None:
        with self._configured():
            super().delete(event_id)

    def list_upcoming(self, start: datetime, end: datetime) -> list[Event]:
        if start.utcoffset() is None or end.utcoffset() is None or start >= end:
            raise CalendarError(
                ErrorCode.INVALID_INPUT,
                "start and end must be timezone-aware, with start before end.",
                retryable=False,
            )
        with self._configured(), self._errors("list"):
            return self._events.list_events(start.isoformat(), end.isoformat())


def create_calendar_service(
    *,
    token_file: str | Path | None = None,
    credentials: Credentials | None = None,
) -> UpcomingCalendarService:
    """Bind one OAuth grant; authorize lazily on the first call.

    Pass exactly one: an external token file, or credentials held in memory.
    Refresh stays in memory. With credentials, it updates that object, so the
    caller can save the new access token and expiry afterwards.
    """
    if token_file is not None and credentials is None:
        return _ConfiguredCalendarService(
            TokenAuthorization(token_file), _TOKEN_FILE_SETUP
        )
    if credentials is not None and token_file is None:
        if not isinstance(credentials, Credentials):
            raise TypeError("credentials must be google.oauth2.credentials.Credentials")
        return _ConfiguredCalendarService(
            CredentialsAuthorization(credentials), _CREDENTIALS_SETUP
        )
    raise ValueError("Pass exactly one of token_file or credentials.")
