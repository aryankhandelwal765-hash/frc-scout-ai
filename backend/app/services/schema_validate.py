"""Validate scouting/pre-scout payloads against a season's JSON Schema.

Unknown fields are always rejected: every object node in the schema is closed
(additionalProperties=false) unless the schema author set it explicitly.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class PayloadInvalid(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def _close(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
        for v in node.values():
            _close(v)
    elif isinstance(node, list):
        for v in node:
            _close(v)


def compile_schema(schema: str | dict) -> Draft202012Validator:
    parsed = json.loads(schema) if isinstance(schema, str) else copy.deepcopy(schema)
    if not isinstance(parsed, dict):
        raise SchemaError("schema must be a JSON object")
    parsed.setdefault("type", "object")
    _close(parsed)
    Draft202012Validator.check_schema(parsed)
    return Draft202012Validator(parsed)


def validate_payload(schema: str | dict | None, data: Any) -> None:
    """Raise PayloadInvalid with readable messages, or return None when valid."""
    if schema is None:
        raise PayloadInvalid(["no schema defined for this season"])
    validator = compile_schema(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        raise PayloadInvalid(
            [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}" for e in errors]
        )
