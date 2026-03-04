"""Translate gRPC filter messages to Qdrant filter conditions."""

import json

from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)


def _is_date_string(value) -> bool:
    """Check if a value looks like a date/datetime string."""
    return isinstance(value, str) and len(value) >= 10 and value[4:5] == "-"


def _make_range(gte=None, lte=None):
    """Create a Range or DatetimeRange depending on value types."""
    if (gte is not None and _is_date_string(gte)) or (lte is not None and _is_date_string(lte)):
        return DatetimeRange(gte=gte, lte=lte)
    return Range(gte=gte, lte=lte)


def build_qdrant_filter(grpc_filters: list) -> Filter | None:
    """Convert a list of gRPC Filter messages to a Qdrant Filter.

    Supported operators: eq, in, gte, lte, range
    """
    if not grpc_filters:
        return None

    conditions = []
    for f in grpc_filters:
        field = f.field
        operator = f.operator
        value = json.loads(f.value)

        match operator:
            case "eq":
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
            case "in":
                conditions.append(
                    FieldCondition(key=field, match=MatchAny(any=value))
                )
            case "gte":
                conditions.append(
                    FieldCondition(key=field, range=_make_range(gte=value))
                )
            case "lte":
                conditions.append(
                    FieldCondition(key=field, range=_make_range(lte=value))
                )
            case "range":
                conditions.append(
                    FieldCondition(
                        key=field,
                        range=_make_range(gte=value.get("gte"), lte=value.get("lte")),
                    )
                )
            case _:
                raise ValueError(f"Unsupported filter operator: {operator}")

    return Filter(must=conditions)
