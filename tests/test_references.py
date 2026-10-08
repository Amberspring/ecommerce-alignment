import pytest
from ecom.evaluate import score_references


def test_reference_overlap_not_keyword_or_truncated_success():
    rows = [{"id": "a", "references": ["The blue case", "blue cover"]}]
    assert score_references(rows, [{"id": "a", "answer": "a blue case"}])["reference_exact_match_rate"] == 1
    assert score_references(rows, [{"id": "a", "answer": "blue case", "finish_reason": "length"}])["mean_reference_token_f1"] == 0
    assert score_references(rows, [{"id": "a", "answer": "red case"}])["mean_reference_token_f1"] == .5
    with pytest.raises(ValueError):
        score_references(rows, [{"id": "b", "answer": "blue case"}])
