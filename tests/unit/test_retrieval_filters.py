"""Unit tests for retrieval service filter builder."""

import json
from datetime import datetime

import pytest
from qdrant_client.models import DatetimeRange, FieldCondition, MatchAny, MatchValue, Range

from services.retrieval.src.search.filters import build_qdrant_filter


class MockFilter:
    """Mock gRPC Filter message."""

    def __init__(self, field: str, operator: str, value):
        self.field = field
        self.operator = operator
        self.value = json.dumps(value)


class TestBuildQdrantFilter:
    def test_empty_filters_returns_none(self):
        assert build_qdrant_filter([]) is None

    def test_eq_filter(self):
        filters = [MockFilter("team", "eq", "platform")]
        result = build_qdrant_filter(filters)
        assert result is not None
        assert len(result.must) == 1
        cond = result.must[0]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "team"
        assert isinstance(cond.match, MatchValue)
        assert cond.match.value == "platform"

    def test_in_filter(self):
        filters = [MockFilter("tags", "in", ["runbook", "auth"])]
        result = build_qdrant_filter(filters)
        assert result is not None
        cond = result.must[0]
        assert isinstance(cond.match, MatchAny)
        assert cond.match.any == ["runbook", "auth"]

    def test_gte_filter_datetime(self):
        filters = [MockFilter("ingested_at", "gte", "2026-01-01")]
        result = build_qdrant_filter(filters)
        cond = result.must[0]
        assert isinstance(cond.range, DatetimeRange)
        assert cond.range.gte == datetime(2026, 1, 1)

    def test_gte_filter_numeric(self):
        filters = [MockFilter("score", "gte", 0.5)]
        result = build_qdrant_filter(filters)
        cond = result.must[0]
        assert isinstance(cond.range, Range)
        assert cond.range.gte == 0.5

    def test_lte_filter_datetime(self):
        filters = [MockFilter("ingested_at", "lte", "2026-12-31")]
        result = build_qdrant_filter(filters)
        cond = result.must[0]
        assert isinstance(cond.range, DatetimeRange)
        assert cond.range.lte == datetime(2026, 12, 31)

    def test_range_filter_datetime(self):
        filters = [MockFilter("ingested_at", "range", {"gte": "2026-01-01", "lte": "2026-12-31"})]
        result = build_qdrant_filter(filters)
        cond = result.must[0]
        assert isinstance(cond.range, DatetimeRange)
        assert cond.range.gte == datetime(2026, 1, 1)
        assert cond.range.lte == datetime(2026, 12, 31)

    def test_multiple_filters(self):
        filters = [
            MockFilter("team", "eq", "platform"),
            MockFilter("tags", "in", ["runbook"]),
        ]
        result = build_qdrant_filter(filters)
        assert len(result.must) == 2

    def test_unsupported_operator_raises(self):
        filters = [MockFilter("team", "regex", "plat.*")]
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            build_qdrant_filter(filters)
