import json

import pytest

from app.services.schema_validate import PayloadInvalid, compile_schema, validate_payload
from app.services.seasons import EXAMPLE_PRESCOUT_SCHEMA, EXAMPLE_SCOUTING_SCHEMA

GOOD = {"auto_scored": 3, "teleop_scored": 10, "driver_skill": 4}


def test_valid_payload_passes():
    validate_payload(json.dumps(EXAMPLE_SCOUTING_SCHEMA), GOOD)


def test_unknown_field_rejected():
    with pytest.raises(PayloadInvalid) as e:
        validate_payload(EXAMPLE_SCOUTING_SCHEMA, {**GOOD, "sneaky": 1})
    assert "sneaky" in str(e.value)


def test_unknown_field_rejected_even_if_schema_omits_additional_properties():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    with pytest.raises(PayloadInvalid):
        validate_payload(schema, {"a": 1, "b": 2})


def test_nested_objects_are_closed_too():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "object", "properties": {"x": {"type": "integer"}}}},
    }
    validate_payload(schema, {"n": {"x": 1}})
    with pytest.raises(PayloadInvalid):
        validate_payload(schema, {"n": {"x": 1, "y": 2}})


def test_type_range_required_enforced():
    for bad in (
        {"auto_scored": "3", "teleop_scored": 1},
        {"auto_scored": -1, "teleop_scored": 1},
        {"auto_scored": 1},
        {"auto_scored": 1, "teleop_scored": 1, "driver_skill": 9},
    ):
        with pytest.raises(PayloadInvalid):
            validate_payload(EXAMPLE_SCOUTING_SCHEMA, bad)


def test_missing_schema_and_bad_schema():
    with pytest.raises(PayloadInvalid):
        validate_payload(None, {})
    with pytest.raises(Exception):  # noqa: B017 - SchemaError
        compile_schema({"type": "not-a-type"})


def test_prescout_schema():
    validate_payload(EXAMPLE_PRESCOUT_SCHEMA, {"drivetrain": "swerve", "reliability": 3})
    with pytest.raises(PayloadInvalid):
        validate_payload(EXAMPLE_PRESCOUT_SCHEMA, {"drivetrain": "hovercraft"})


def test_example_season_is_clearly_labelled():
    from app.services.seasons import EXAMPLE_GAME_NAME

    assert "EXAMPLE" in EXAMPLE_GAME_NAME and "replace me" in EXAMPLE_GAME_NAME
