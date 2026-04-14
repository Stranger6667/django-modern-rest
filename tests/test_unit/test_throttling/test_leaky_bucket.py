import json
from http import HTTPStatus
from typing import Final, TypeAlias

import pytest
from django.http import HttpResponse
from freezegun.api import FrozenDateTimeFactory
from inline_snapshot import snapshot

from dmr import Controller, modify
from dmr.plugins.pydantic import PydanticSerializer
from dmr.serializer import BaseSerializer
from dmr.test import DMRAsyncRequestFactory, DMRRequestFactory
from dmr.throttling import AsyncThrottle, Rate, SyncThrottle
from dmr.throttling.algorithms import LeakyBucket

_Serializes: TypeAlias = list[type[BaseSerializer]]
serializers: Final[_Serializes] = [
    PydanticSerializer,
]

try:
    from dmr.plugins.msgspec import MsgspecSerializer
except ImportError:  # pragma: no cover
    pass  # noqa: WPS420
else:  # pragma: no cover
    serializers.append(MsgspecSerializer)


@pytest.mark.parametrize('serializer', serializers)
def test_leaky_bucket_sync_fill_and_reject(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
    *,
    serializer: type[BaseSerializer],
) -> None:
    """Fill the bucket to capacity, then get rejected."""

    class _Controller(
        Controller[serializer],  # type: ignore[valid-type]
    ):
        @modify(
            throttling=[
                SyncThrottle(
                    2,
                    10,
                    algorithm=LeakyBucket(),
                ),
            ],
        )
        def get(self) -> str:
            return 'ok'

    # Two requests fill the bucket:
    for _ in range(2):
        request = dmr_rf.get('/whatever/')
        response = _Controller.as_view()(request)
        assert isinstance(response, HttpResponse)
        assert response.status_code == HTTPStatus.OK

    # Third is rejected:
    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert response.headers == {
        'X-RateLimit-Limit': '2',
        'X-RateLimit-Remaining': '0',
        'X-RateLimit-Reset': '10',
        'Retry-After': '10',
        'Content-Type': 'application/json',
    }
    assert json.loads(response.content) == snapshot({
        'detail': [
            {'msg': 'Too many requests', 'type': 'ratelimit'},
        ],
    })


@pytest.mark.parametrize('serializer', serializers)
def test_leaky_bucket_smooth_drain(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
    *,
    serializer: type[BaseSerializer],
) -> None:
    """Verify smooth draining: partial time frees partial capacity."""

    class _Controller(
        Controller[serializer],  # type: ignore[valid-type]
    ):
        @modify(
            throttling=[
                SyncThrottle(
                    2,
                    10,
                    algorithm=LeakyBucket(),
                ),
            ],
        )
        def get(self) -> str:
            return 'ok'

    # Fill the bucket:
    for _ in range(2):
        request = dmr_rf.get('/whatever/')
        response = _Controller.as_view()(request)
        assert response.status_code == HTTPStatus.OK

    # Rejected while full:
    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    # After 5 seconds one token leaks (leak_interval=10/2=5):
    freezer.tick(delta=5)

    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK

    # Bucket is full again - immediately rejected:
    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS


@pytest.mark.parametrize('serializer', serializers)
def test_leaky_bucket_full_drain(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
    *,
    serializer: type[BaseSerializer],
) -> None:
    """After full duration the bucket is empty again."""

    class _Controller(
        Controller[serializer],  # type: ignore[valid-type]
    ):
        @modify(
            throttling=[
                SyncThrottle(
                    2,
                    10,
                    algorithm=LeakyBucket(),
                ),
            ],
        )
        def get(self) -> str:
            return 'ok'

    # Fill the bucket:
    for _ in range(2):
        request = dmr_rf.get('/whatever/')
        response = _Controller.as_view()(request)
        assert response.status_code == HTTPStatus.OK

    # After full duration everything drains:
    freezer.tick(delta=10)

    for _ in range(2):
        request = dmr_rf.get('/whatever/')
        response = _Controller.as_view()(request)
        assert isinstance(response, HttpResponse)
        assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize(
    'rate',
    [Rate.second, Rate.minute, Rate.hour],
)
def test_leaky_bucket_rates(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
    *,
    rate: Rate,
) -> None:
    """Rates work correctly with the leaky bucket."""

    class _Controller(Controller[PydanticSerializer]):
        throttling = [
            SyncThrottle(1, rate, algorithm=LeakyBucket()),
        ]

        def get(self) -> str:
            return 'ok'

    # First is ok:
    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK

    # Second is rate limited:
    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    # After full duration, it is ok:
    freezer.tick(delta=int(rate))

    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
@pytest.mark.parametrize('serializer', serializers)
async def test_leaky_bucket_async(
    dmr_async_rf: DMRAsyncRequestFactory,
    freezer: FrozenDateTimeFactory,
    *,
    serializer: type[BaseSerializer],
) -> None:
    """Async controllers work with the leaky bucket algorithm."""

    class _AsyncController(
        Controller[serializer],  # type: ignore[valid-type]
    ):
        throttling = [
            AsyncThrottle(2, 10, algorithm=LeakyBucket()),
        ]

        async def get(self) -> str:
            return 'ok'

    # Fill the bucket:
    for _ in range(2):
        request = dmr_async_rf.get('/whatever/')
        response = await dmr_async_rf.wrap(  # noqa: WPS476
            _AsyncController.as_view()(request),
        )
        assert isinstance(response, HttpResponse)
        assert response.status_code == HTTPStatus.OK

    # Rejected:
    request = dmr_async_rf.get('/whatever/')
    response = await dmr_async_rf.wrap(
        _AsyncController.as_view()(request),
    )
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    # After partial drain one more is allowed:
    freezer.tick(delta=5)

    request = dmr_async_rf.get('/whatever/')
    response = await dmr_async_rf.wrap(
        _AsyncController.as_view()(request),
    )
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize('serializer', serializers)
def test_leaky_bucket_per_endpoint_isolation(
    dmr_rf: DMRRequestFactory,
    freezer: FrozenDateTimeFactory,
    *,
    serializer: type[BaseSerializer],
) -> None:
    """Different endpoints have separate buckets."""

    class _Controller(
        Controller[serializer],  # type: ignore[valid-type]
    ):
        @modify(
            throttling=[
                SyncThrottle(
                    1,
                    Rate.second,
                    algorithm=LeakyBucket(),
                ),
            ],
        )
        def get(self) -> str:
            return 'get_ok'

        @modify(
            throttling=[
                SyncThrottle(
                    1,
                    Rate.second,
                    algorithm=LeakyBucket(),
                ),
            ],
        )
        def put(self) -> str:
            return 'put_ok'

    # Fill GET bucket:
    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert response.status_code == HTTPStatus.OK

    # GET is now rejected:
    request = dmr_rf.get('/whatever/')
    response = _Controller.as_view()(request)
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    # PUT still works (separate bucket):
    request = dmr_rf.put('/whatever/')
    response = _Controller.as_view()(request)
    assert isinstance(response, HttpResponse)
    assert response.status_code == HTTPStatus.OK
