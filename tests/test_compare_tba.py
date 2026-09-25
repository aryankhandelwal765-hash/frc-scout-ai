"""compare_tba.dotted — nested lookup into TBA's score_breakdown."""

from compare_tba import dotted, tba_value


def test_single_hop():
    assert dotted({"a": 1}, "a") == 1


def test_nested_hop():
    assert dotted({"a": {"b": {"c": 7}}}, "a.b.c") == 7


def test_missing_leaf_is_none():
    assert dotted({"a": {"b": 1}}, "a.zzz") is None


def test_missing_intermediate_is_none():
    assert dotted({"a": 1}, "a.b.c") is None


def test_non_dict_midway_is_none_not_a_typeerror():
    """TBA sometimes hands back a scalar where a dict is expected."""
    assert dotted({"a": "string"}, "a.b") is None
    assert dotted({"a": [1, 2]}, "a.b") is None


def test_empty_mapping():
    assert dotted({}, "anything") is None


def test_falsy_values_survive():
    """0 and False are real readings, not misses."""
    assert dotted({"a": {"b": 0}}, "a.b") == 0
    assert dotted({"a": {"b": False}}, "a.b") is False


def test_none_value_is_returned_not_confused_with_missing():
    assert dotted({"a": {"b": None}}, "a.b") is None


def test_tba_value_none_field_reads_the_alliance_score():
    truth = {"red_score": 120, "blue_score": 98, "score_breakdown": {}}
    assert tba_value(truth, "red", None) == 120
    assert tba_value(truth, "blue", None) == 98


def test_tba_value_dotted_field_reads_the_breakdown():
    truth = {
        "red_score": 120,
        "score_breakdown": {"red": {"hubScore": {"totalCount": 42}}},
    }
    assert tba_value(truth, "red", "hubScore.totalCount") == 42


def test_tba_value_missing_breakdown_is_none():
    assert tba_value({"score_breakdown": {}}, "red", "hubScore.totalCount") is None
