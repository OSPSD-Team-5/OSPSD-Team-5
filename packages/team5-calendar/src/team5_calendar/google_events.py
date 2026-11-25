"""Translate Google events to the public timed-event model."""

from collections.abc import Mapping
from contextlib import closing

from calendar_contract import Event
from google.auth.exceptions import TransportError
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from httplib2 import HttpLib2Error  # type: ignore[import-untyped]
from pydantic import ValidationError
from requests.exceptions import RequestException

from .event_provider import (
    CalendarAuthenticationError,
    CalendarProviderError,
    EventAccessDeniedError,
    EventNotFoundError,
    EventReadUnavailableError,
    UnsupportedEventError,
)
from .google_auth import calendar_client
from .tracing import google_call

# Google can report quota/rate limits as 403 instead of 429.
RATE_LIMIT_REASONS = {"rateLimitExceeded", "userRateLimitExceeded", "quotaExceeded"}
TEMPORARY_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}


def is_rate_limited(error: HttpError) -> bool:
    """Recognize Google's structured limit reasons without using error text."""
    details = error.error_details
    return isinstance(details, list) and any(
        isinstance(detail, dict)
        and isinstance(detail.get("reason"), str)
        and detail["reason"] in RATE_LIMIT_REASONS
        for detail in details
    )


def translate_event(data: Mapping[str, object]) -> Event:
    start = data.get("start")
    end = data.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise CalendarProviderError
    if "dateTime" not in start or "dateTime" not in end:
        if "date" in start or "date" in end:
            raise UnsupportedEventError
        raise CalendarProviderError
    # Google permits untitled events; preserve them as an empty title.
    try:
        return Event.model_validate(
            {
                "id": data.get("id"),
                "title": data.get("summary", ""),
                "start": start["dateTime"],
                "end": end["dateTime"],
            }
        )
    except ValidationError as error:
        raise CalendarProviderError from error


class GoogleEventReader:
    """Read Google events through the service event contract."""

    def get_event(self, event_id: str) -> Event:
        try:
            with closing(calendar_client()) as service:
                with google_call("events.get"):
                    data = (
                        service.events()
                        .get(calendarId="primary", eventId=event_id)
                        .execute()
                    )
        except HttpError as error:
            if error.resp.status in (404, 410):
                raise EventNotFoundError from error
            raise _read_failure(error) from error
        except (TransportError, HttpLib2Error, RequestException, OSError) as error:
            raise EventReadUnavailableError from error
        if not isinstance(data, dict):
            raise CalendarProviderError
        if data.get("status") == "cancelled":
            raise EventNotFoundError
        return translate_event(data)

    def list_events(self, time_min: str, time_max: str) -> list[Event]:
        """Return timed events overlapping time_min to time_max, earliest first.

        Recurring events are expanded into instances. All-day and cancelled
        events are skipped. Raise the same errors as get_event, except that
        nothing is ever not found.
        """
        pages: list[object] = []
        page_token: str | None = None
        try:
            with closing(calendar_client()) as service:
                while True:
                    with google_call("events.list"):
                        page = (
                            service.events()
                            .list(
                                calendarId="primary",
                                timeMin=time_min,
                                timeMax=time_max,
                                singleEvents=True,
                                orderBy="startTime",
                                maxResults=250,
                                pageToken=page_token,
                            )
                            .execute()
                        )
                    pages.append(page)
                    token = (
                        page.get("nextPageToken") if isinstance(page, dict) else None
                    )
                    if not isinstance(token, str) or not token:
                        break
                    page_token = token
        except HttpError as error:
            raise _read_failure(error) from error
        except (TransportError, HttpLib2Error, RequestException, OSError) as error:
            raise EventReadUnavailableError from error
        return [event for page in pages for event in _timed_events(page)]


def _read_failure(error: HttpError) -> Exception:
    """Map a failed Google read, other than a missing event, to a reader error."""
    status = error.resp.status
    if status == 401:
        return CalendarAuthenticationError(provider=True)
    if status in TEMPORARY_HTTP_STATUSES or (status == 403 and is_rate_limited(error)):
        return EventReadUnavailableError()
    if status == 403:
        return EventAccessDeniedError()
    return CalendarProviderError()


def _timed_events(page: object) -> list[Event]:
    items = page.get("items", []) if isinstance(page, dict) else None
    if not isinstance(items, list):
        raise CalendarProviderError
    events: list[Event] = []
    for item in items:
        if not isinstance(item, dict):
            raise CalendarProviderError
        if item.get("status") == "cancelled":
            continue
        try:
            events.append(translate_event(item))
        except UnsupportedEventError:
            continue  # all-day: no instant to return
    return events
