from ecom.data import prepare, norm, validate_preferences
from ecom.evaluate import score
import pytest
from pathlib import Path


def test_redaction_and_prompt_split():
    rows = [{"prompt": f"问题{i}", "completion": "回复"} for i in range(10)]
    rows += [rows[0], {"prompt": "", "completion": "x"}]
    s = prepare(rows)
    groups = [{r["group_id"] for r in s[k]} for k in ("train", "validation", "test")]
    assert (
        not groups[0] & groups[1]
        and not groups[0] & groups[2]
        and not groups[1] & groups[2]
    )
    assert len(s["rejected"]) == 2
    assert norm("电话13812345678 a@b.com") == "电话[PHONE] [EMAIL]"


def test_preference_no_leakage():
    rows = [
        dict(prompt="train", chosen="yes", rejected="no"),
        dict(prompt="test", chosen="yes", rejected="no"),
    ]
    assert len(validate_preferences(rows, {"train"})) == 1
    with pytest.raises(ValueError):
        validate_preferences([dict(prompt="x", chosen="same", rejected="same")])


def test_missing_predictions_rejected():
    with pytest.raises(ValueError):
        score([dict(id="a", must_include=["x"])], [])


def test_policy_group_split():
    rows = [dict(prompt=f"policy{i} question{j}", completion="answer", policy_id=f"p{i}") for i in range(10) for j in range(3)]
    splits = prepare(rows)
    groups = [{r["policy_id"] for r in splits[k]} for k in ("train", "validation", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert sum(len(splits[k]) for k in ("train", "validation", "test")) == 30


def test_rubric_and_badcase():
    r = score(
        [dict(id="a", must_include=["退款"], must_not_include=["保证"])],
        [dict(id="a", answer="保证退款")],
    )
    assert r["rubric_pass_rate"] == 0 and r["badcases"]


def test_jsonl_bytes_are_stable_across_platforms(tmp_path):
    from ecom.data import write

    target = tmp_path / "rows.jsonl"
    write(target, [{"id": "a"}, {"id": "b"}])
    assert target.read_bytes() == b'{"id": "a"}\r\n{"id": "b"}\r\n'
