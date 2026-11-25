"""Consumer-facing Calendar factory checks with controlled Google transport."""

import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock, Mock

import pytest
from calendar_contract import (
    CalendarError,
    CalendarService,
    CreateEvent,
    ErrorCode,
    Event,
    ReplaceEvent,
)
from google.auth.exceptions import RefreshError, TransportError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from team5_calendar import (
    authorization,
    authorize,
    create_calendar_service,
    google_auth,
    google_create_events,
    google_events,
)

from tests.test_shared_conformance import (
    TestGoogleCalendarServiceConformance as _GoogleConformance,
)

PAYLOAD = {
    "title": "Package consumer check",
    "start": "2030-01-07T15:00:00Z",
    "end": "2030-01-07T15:30:00Z",
}


def test_factory_is_lazy_and_does_not_need_a_team5_app(tmp_path: Path) -> None:
    service = create_calendar_service(token_file=tmp_path / "not-yet-authorized.json")
    assert callable(service.create)
    assert callable(service.get)
    assert callable(service.replace)
    assert callable(service.delete)
    assert list(tmp_path.iterdir()) == []


def test_factory_runs_existing_google_crud_and_cleans_up(
    google_service: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(google_events, "calendar_client", lambda: google_service)
    service = create_calendar_service(token_file=tmp_path / "unused-test-token.json")

    created = service.create(CreateEvent(**PAYLOAD))
    assert created.model_dump() == {"id": "google-event-1", **PAYLOAD}
    assert service.get(created.id) == created
    replacement = ReplaceEvent(**{**PAYLOAD, "title": "Updated package event"})
    updated = service.replace(created.id, replacement)
    assert updated.id == created.id
    assert updated.title == replacement.title
    assert service.get(created.id) == updated
    service.delete(created.id)
    with pytest.raises(CalendarError) as caught:
        service.get(created.id)
    assert caught.value.code == ErrorCode.NOT_FOUND
    with pytest.raises(CalendarError) as caught:
        service.delete(created.id)
    assert caught.value.code == ErrorCode.NOT_FOUND
    google_service.events.return_value.insert.return_value.execute.assert_called_once_with(
        num_retries=0
    )


def test_factory_preserves_uncertain_create_without_retry(
    google_service: MagicMock, tmp_path: Path
) -> None:
    execute = google_service.events.return_value.insert.return_value.execute
    execute.side_effect = TimeoutError("private provider detail")
    service = create_calendar_service(token_file=tmp_path / "unused-test-token.json")
    with pytest.raises(CalendarError) as caught:
        service.create(CreateEvent(**PAYLOAD))
    assert caught.value.code == ErrorCode.PROVIDER_ERROR
    assert caught.value.retryable is False
    assert "private provider detail" not in caught.value.message
    execute.assert_called_once_with(num_retries=0)
    google_service.events.return_value.delete.assert_not_called()


def test_missing_external_token_uses_shared_error(tmp_path: Path) -> None:
    service = create_calendar_service(token_file=tmp_path / "missing.json")
    with pytest.raises(CalendarError) as caught:
        service.get("event-1")
    assert caught.value.code == ErrorCode.PROVIDER_UNAVAILABLE
    assert caught.value.retryable is False
    assert str(tmp_path) not in caught.value.message
    assert authorization.active_authorization.get() is None


def test_explicit_token_paths_are_isolated_and_override_host_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = [tmp_path / "account-a.json", tmp_path / "account-b.json"]
    for path in paths:
        path.write_text("{}", encoding="utf-8")
    credentials = [Mock(valid=True), Mock(valid=True)]
    for credential in credentials:
        credential.has_scopes.return_value = True
    load = Mock(side_effect=credentials)
    monkeypatch.setattr(Credentials, "from_authorized_user_file", load)
    monkeypatch.setenv("K_SERVICE", "another-teams-service")
    monkeypatch.setenv("CALENDAR_MODE", "hosted")
    monkeypatch.setenv("CALENDAR_TOKEN_FILE", str(tmp_path / "wrong-account.json"))

    selected: list[object] = []
    sdk = MagicMock()
    sdk.events.return_value.get.return_value.execute.return_value = {
        "id": "event-1",
        "summary": PAYLOAD["title"],
        "start": {"dateTime": PAYLOAD["start"]},
        "end": {"dateTime": PAYLOAD["end"]},
    }

    def build(api: str, version: str, *, credentials: object) -> MagicMock:
        assert (api, version) == ("calendar", "v3")
        selected.append(credentials)
        return sdk

    monkeypatch.setattr(google_auth, "build", build)
    first = create_calendar_service(token_file=paths[0])
    second = create_calendar_service(token_file=paths[1])
    assert first.get("event-1").id == "event-1"
    assert second.get("event-1").id == "event-1"
    assert first.get("event-1").id == "event-1"
    assert selected == [credentials[0], credentials[1], credentials[0]]
    assert [call.args[0] for call in load.call_args_list] == [str(p) for p in paths]


def test_external_refresh_stays_in_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "token.json"
    path.write_text("unchanged private fixture", encoding="utf-8")
    credential = Mock(valid=False, expired=True, refresh_token="private-refresh")
    credential.has_scopes.return_value = True

    def refresh(request: object) -> None:
        credential.valid = True

    credential.refresh.side_effect = refresh
    monkeypatch.setattr(
        Credentials,
        "from_authorized_user_file",
        Mock(return_value=credential),
    )
    configured = authorization.TokenAuthorization(path)
    assert configured.load() is credential
    assert configured.load() is credential
    credential.refresh.assert_called_once()
    credential.to_json.assert_not_called()
    assert path.read_text(encoding="utf-8") == "unchanged private fixture"


def test_factory_context_is_isolated_between_concurrent_consumers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    barrier = Barrier(2)
    paths = [tmp_path / "a.json", tmp_path / "b.json"]

    def read(self: object, event_id: str) -> Event:
        barrier.wait(timeout=5)
        configured = authorization.active_authorization.get()
        assert isinstance(configured, authorization.TokenAuthorization)
        return Event(id=event_id, **{**PAYLOAD, "title": configured.path.name})

    monkeypatch.setattr(google_events.GoogleEventReader, "get_event", read)
    services = [create_calendar_service(token_file=path) for path in paths]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda service: service.get("event"), services))
    assert [event.title for event in results] == [path.name for path in paths]
    assert authorization.active_authorization.get() is None


def test_explicit_authorize_command_writes_private_external_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "external-token.json"
    client = tmp_path / "client.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "authorize",
            "--client-secrets",
            str(client),
            "--token-file",
            str(destination),
        ],
    )
    flow = Mock()
    flow.run_local_server.return_value.has_scopes.return_value = True
    flow.run_local_server.return_value.to_json.return_value = '{"token":"synthetic"}'
    create = Mock(return_value=flow)
    monkeypatch.setattr(InstalledAppFlow, "from_client_secrets_file", create)
    open_token = Mock(wraps=os.open)
    monkeypatch.setattr(os, "open", open_token)
    assert authorize.main() == 0
    create.assert_called_once_with(str(client), authorization.SCOPES)
    flow.run_local_server.assert_called_once_with(
        port=0, timeout_seconds=300, authorization_prompt_message=""
    )
    assert destination.read_text(encoding="utf-8") == '{"token":"synthetic"}'
    open_token.assert_called_once_with(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    # Windows mode bits do not represent POSIX owner-only permissions.
    if os.name == "posix":
        assert destination.stat().st_mode & 0o777 == 0o600


def test_authorize_command_does_not_replace_an_existing_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "external-token.json"
    destination.write_text("existing grant", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "authorize",
            "--client-secrets",
            str(tmp_path / "client.json"),
            "--token-file",
            str(destination),
        ],
    )
    flow = Mock()
    flow.run_local_server.return_value.has_scopes.return_value = True
    monkeypatch.setattr(
        InstalledAppFlow, "from_client_secrets_file", Mock(return_value=flow)
    )
    assert authorize.main() == 1
    assert destination.read_text(encoding="utf-8") == "existing grant"


def test_authorize_command_sanitizes_setup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "external-token.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "authorize",
            "--client-secrets",
            str(tmp_path / "client.json"),
            "--token-file",
            str(destination),
        ],
    )
    monkeypatch.setattr(
        InstalledAppFlow,
        "from_client_secrets_file",
        Mock(side_effect=ValueError("SECRET client details")),
    )
    assert authorize.main() == 1
    assert not destination.exists()
    assert "SECRET" not in capsys.readouterr().out


class TestPackageConformance(_GoogleConformance):
    """Run the unchanged shared suite through the public consumer factory."""

    @pytest.fixture
    def service(
        self,
        google_service: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> CalendarService:
        monkeypatch.setattr(google_events, "calendar_client", lambda: google_service)
        return create_calendar_service(token_file=tmp_path / "controlled-token.json")


@pytest.mark.parametrize("contents", ["[]", "null", "42", '"private-token"', "{bad"])
def test_malformed_token_is_a_sanitized_shared_error(
    tmp_path: Path, contents: str
) -> None:
    token = tmp_path / "token.json"
    token.write_text(contents)
    service = create_calendar_service(token_file=token)
    with pytest.raises(CalendarError) as raised:
        service.get("synthetic-event-id")
    assert raised.value.code == ErrorCode.PROVIDER_UNAVAILABLE
    assert raised.value.retryable is False
    assert "private-token" not in str(raised.value)
    assert "team5_calendar.authorize" in str(raised.value)


def _credentials(
    user: str, *, access: bool = True, scopes: list[str] = authorization.SCOPES
) -> Credentials:
    """A grant as a consumer rebuilds it from its own encrypted store."""
    return Credentials(
        token=f"{user}-access" if access else None,
        refresh_token=f"{user}-refresh",
        client_id="client-id.apps.googleusercontent.com",
        client_secret="private-client-secret",
        token_uri="https://oauth2.googleapis.com/token",
        scopes=scopes,
    )


def _record_reads(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, list[object]]:
    """Fake the SDK behind get and list; record which credentials built it."""
    selected: list[object] = []
    sdk = MagicMock()
    sdk.events.return_value.get.return_value.execute.return_value = {
        "id": "event-1",
        "summary": PAYLOAD["title"],
        "start": {"dateTime": PAYLOAD["start"]},
        "end": {"dateTime": PAYLOAD["end"]},
    }
    sdk.events.return_value.list.return_value.execute.return_value = {"items": []}

    def build(api: str, version: str, *, credentials: object) -> MagicMock:
        assert (api, version) == ("calendar", "v3")
        selected.append(credentials)
        return sdk

    monkeypatch.setattr(google_auth, "build", build)
    return sdk, selected


WINDOW = (datetime(2030, 1, 7, tzinfo=UTC), datetime(2030, 1, 21, tzinfo=UTC))


def test_factory_takes_exactly_one_grant(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        create_calendar_service()
    with pytest.raises(ValueError):
        create_calendar_service(
            token_file=tmp_path / "token.json", credentials=_credentials("a")
        )
    with pytest.raises(TypeError):
        create_calendar_service(credentials={"refresh_token": "a"})  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


def test_credentials_services_act_only_as_their_own_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("K_SERVICE", "another-teams-service")
    monkeypatch.setenv("CALENDAR_MODE", "hosted")
    monkeypatch.setenv("CALENDAR_TOKEN_FILE", str(tmp_path / "wrong-account.json"))
    read_file = Mock(side_effect=AssertionError("no token file is read"))
    monkeypatch.setattr(Credentials, "from_authorized_user_file", read_file)
    _, selected = _record_reads(monkeypatch)
    created_with: list[object] = []
    writer = MagicMock()
    insert = writer.__enter__.return_value.events.return_value.insert
    insert.return_value.execute.return_value = {
        "id": "event-2",
        "summary": PAYLOAD["title"],
        "start": {"dateTime": PAYLOAD["start"]},
        "end": {"dateTime": PAYLOAD["end"]},
    }

    def build_writer(api: str, version: str, *, credentials: object) -> MagicMock:
        created_with.append(credentials)
        return writer

    monkeypatch.setattr(google_create_events, "build", build_writer)
    user_a, user_b = _credentials("a"), _credentials("b")
    first = create_calendar_service(credentials=user_a)
    second = create_calendar_service(credentials=user_b)

    assert first.get("event-1").id == "event-1"
    assert second.get("event-1").id == "event-1"
    assert second.list_upcoming(*WINDOW) == []
    assert first.get("event-1").id == "event-1"
    assert second.create(CreateEvent(**PAYLOAD)).id == "event-2"
    assert selected == [user_a, user_b, user_b, user_a]
    assert created_with == [user_b]
    read_file.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_credentials_context_is_isolated_between_concurrent_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = Barrier(2)

    def read(self: object, event_id: str) -> Event:
        barrier.wait(timeout=5)
        configured = authorization.active_authorization.get()
        assert configured is not None
        return Event(id=event_id, **{**PAYLOAD, "title": str(configured.load().token)})

    monkeypatch.setattr(google_events.GoogleEventReader, "get_event", read)
    services = [create_calendar_service(credentials=_credentials(u)) for u in "ab"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda service: service.get("event"), services))
    assert [event.title for event in results] == ["a-access", "b-access"]
    assert authorization.active_authorization.get() is None


def test_refresh_updates_the_callers_credentials_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grant = _credentials("a", access=False)
    requests: list[object] = []

    def refresh(self: Credentials, request: object) -> None:
        requests.append(request)
        self.token = "a-refreshed-access"
        self.expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)

    monkeypatch.setattr(Credentials, "refresh", refresh)
    _, selected = _record_reads(monkeypatch)
    service = create_calendar_service(credentials=grant)
    service.get("event-1")
    service.get("event-1")

    assert len(requests) == 1
    assert isinstance(requests[0], partial)
    assert requests[0].keywords == {"timeout": 10}
    assert grant.token == "a-refreshed-access"
    assert grant.expiry is not None
    assert selected == [grant, grant]


def _refresh_raises(error: Exception) -> Callable[[Credentials, object], None]:
    def refresh(self: Credentials, request: object) -> None:
        raise error

    return refresh


@pytest.mark.parametrize(
    ("grant", "refresh_error", "retryable"),
    [
        pytest.param(
            _credentials(
                "a", scopes=["https://www.googleapis.com/auth/calendar.readonly"]
            ),
            None,
            False,
            id="missing-scope",
        ),
        pytest.param(
            Credentials(token=None, scopes=authorization.SCOPES),
            None,
            False,
            id="no-refresh-token",
        ),
        pytest.param(
            _credentials("a", access=False),
            RefreshError("invalid_grant: a-refresh was revoked"),
            False,
            id="revoked",
        ),
        pytest.param(
            _credentials("a", access=False),
            TransportError("network down while sending a-refresh"),
            True,
            id="refresh-unreachable",
        ),
    ],
)
def test_credentials_failures_are_sanitized_shared_errors(
    monkeypatch: pytest.MonkeyPatch,
    grant: Credentials,
    refresh_error: Exception | None,
    retryable: bool,
) -> None:
    if refresh_error is not None:
        monkeypatch.setattr(Credentials, "refresh", _refresh_raises(refresh_error))
    _, selected = _record_reads(monkeypatch)
    service = create_calendar_service(credentials=grant)
    with pytest.raises(CalendarError) as raised:
        service.get("event-1")
    error = raised.value
    assert error.code == ErrorCode.PROVIDER_UNAVAILABLE
    assert error.retryable is retryable
    shown = f"{error} {error.message} {error.details!r}"
    assert "a-refresh" not in shown
    assert "private-client-secret" not in shown
    if not retryable:
        assert "Authorize the account again" in error.message
        assert "token file" not in error.message
    assert selected == []


class TestPackageConformanceWithCredentials(_GoogleConformance):
    """Run the unchanged shared suite on a service built from credentials."""

    @pytest.fixture
    def service(
        self, google_service: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> CalendarService:
        monkeypatch.setattr(google_events, "calendar_client", lambda: google_service)
        return create_calendar_service(credentials=_credentials("conformance"))


def _timed(event_id: str, start: str, end: str, **extra: object) -> dict[str, object]:
    return {
        "id": event_id,
        "summary": f"Review {event_id}",
        "start": {"dateTime": start},
        "end": {"dateTime": end},
        **extra,
    }


def _listing(monkeypatch: pytest.MonkeyPatch, *pages: object) -> MagicMock:
    sdk = MagicMock()
    sdk.events.return_value.list.return_value.execute.side_effect = list(pages)
    monkeypatch.setattr(google_events, "calendar_client", lambda: sdk)
    return sdk


def test_list_upcoming_returns_timed_events_in_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    all_day = {
        "id": "holiday",
        "summary": "Holiday",
        "start": {"date": "2030-01-09"},
        "end": {"date": "2030-01-10"},
    }
    sdk = _listing(
        monkeypatch,
        {
            "items": [
                _timed("a", "2030-01-08T15:00:00Z", "2030-01-08T15:30:00Z"),
                all_day,
                _timed(
                    "gone",
                    "2030-01-09T15:00:00Z",
                    "2030-01-09T16:00:00Z",
                    status="cancelled",
                ),
                _timed("b", "2030-01-10T09:00:00-05:00", "2030-01-10T10:00:00-05:00"),
            ]
        },
    )
    service = create_calendar_service(credentials=_credentials("a"))

    events = service.list_upcoming(*WINDOW)

    assert [event.model_dump() for event in events] == [
        {
            "id": "a",
            "title": "Review a",
            "start": "2030-01-08T15:00:00Z",
            "end": "2030-01-08T15:30:00Z",
        },
        {
            "id": "b",
            "title": "Review b",
            "start": "2030-01-10T09:00:00-05:00",
            "end": "2030-01-10T10:00:00-05:00",
        },
    ]
    sdk.events.return_value.list.assert_called_once_with(
        calendarId="primary",
        timeMin="2030-01-07T00:00:00+00:00",
        timeMax="2030-01-21T00:00:00+00:00",
        singleEvents=True,
        orderBy="startTime",
        maxResults=250,
        pageToken=None,
    )


def test_list_upcoming_follows_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _listing(
        monkeypatch,
        {
            "items": [_timed("a", "2030-01-08T15:00:00Z", "2030-01-08T15:30:00Z")],
            "nextPageToken": "page-2",
        },
        {"items": [_timed("b", "2030-01-09T15:00:00Z", "2030-01-09T15:30:00Z")]},
    )
    service = create_calendar_service(credentials=_credentials("a"))

    assert [event.id for event in service.list_upcoming(*WINDOW)] == ["a", "b"]
    calls = sdk.events.return_value.list.call_args_list
    assert [call.kwargs["pageToken"] for call in calls] == [None, "page-2"]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(datetime(2030, 1, 7), WINDOW[1], id="naive-start"),
        pytest.param(WINDOW[0], datetime(2030, 1, 21), id="naive-end"),
        pytest.param(WINDOW[0], WINDOW[0], id="empty"),
        pytest.param(WINDOW[1], WINDOW[0], id="reversed"),
    ],
)
def test_list_upcoming_rejects_a_bad_window_before_calling_google(
    monkeypatch: pytest.MonkeyPatch, start: datetime, end: datetime
) -> None:
    sdk = _listing(monkeypatch)
    service = create_calendar_service(credentials=_credentials("a"))
    with pytest.raises(CalendarError) as raised:
        service.list_upcoming(start, end)
    assert raised.value.code == ErrorCode.INVALID_INPUT
    assert raised.value.retryable is False
    sdk.events.return_value.list.assert_not_called()


def _http_error(status: int) -> HttpError:
    return HttpError(
        Mock(status=status, reason="private provider reason"),
        b'{"error": {"message": "private provider detail"}}',
    )


@pytest.mark.parametrize(
    ("failure", "code", "retryable"),
    [
        pytest.param(_http_error(401), ErrorCode.PROVIDER_UNAVAILABLE, False, id="401"),
        pytest.param(_http_error(403), ErrorCode.ACCESS_DENIED, False, id="403"),
        pytest.param(_http_error(429), ErrorCode.PROVIDER_UNAVAILABLE, True, id="429"),
        pytest.param(_http_error(503), ErrorCode.PROVIDER_UNAVAILABLE, True, id="503"),
        pytest.param(_http_error(400), ErrorCode.PROVIDER_ERROR, False, id="400"),
        pytest.param(
            TimeoutError("private provider detail"),
            ErrorCode.PROVIDER_UNAVAILABLE,
            True,
            id="timeout",
        ),
        pytest.param(
            {"items": [{"id": "x", "start": {}, "end": {}}]},
            ErrorCode.PROVIDER_ERROR,
            False,
            id="malformed-event",
        ),
        pytest.param(
            "private provider detail", ErrorCode.PROVIDER_ERROR, False, id="not-a-page"
        ),
    ],
)
def test_list_upcoming_failures_are_sanitized_shared_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure: object,
    code: ErrorCode,
    retryable: bool,
) -> None:
    _listing(monkeypatch, failure)
    service = create_calendar_service(credentials=_credentials("a"))
    with pytest.raises(CalendarError) as raised:
        service.list_upcoming(*WINDOW)
    assert raised.value.code == code
    assert raised.value.retryable is retryable
    assert "private" not in f"{raised.value} {raised.value.details!r}"
