import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("alignment", Path(__file__).resolve().parents[1] / "scripts/prepare_amazonqa_alignment.py")
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


def record(answers, label=1):
    return {"qid": 1, "asin": "P1", "questionText": "Does it work?", "review_snippets": ["Popular answer. Preferred option; disliked option. Tie A and tie B."], "answers": answers, "is_answerable": label}


def answer(text, helpful):
    return {"answerText": text, "helpful": helpful}


def test_sft_target_is_selected_only_from_predeclared_human_votes():
    row = record([answer("popular answer", [9, 10]), answer("single answer", [1, 1]), answer("unvoted answer", [0, 0])])
    assert alignment.select_target(row)["text"] == "popular answer"
    assert alignment.select_target(record([answer("unvoted", [0, 0])])) is None
    assert alignment.select_target(record([answer("unsupported white color", [9, 10])])) is None


def test_dpo_requires_two_voted_answers_and_wilson_gap():
    row = record([answer("preferred option", [6, 6]), answer("disliked option", [0, 6])])
    pair = alignment.select_preference(row)
    assert (pair["chosen"], pair["rejected"]) == ("preferred option", "disliked option")
    assert alignment.select_preference(record([answer("weak", [1, 1]), answer("other", [0, 1])])) is None
    assert alignment.select_preference(record([answer("tie A", [3, 3]), answer("tie B", [3, 3])])) is None


def test_answerability_is_separate_from_reference_availability():
    unanswerable = record([answer("community answer still exists", [2, 2])], label=0)
    assert alignment.classify(unanswerable) == "unanswerable"
    assert alignment.voted_answers(unanswerable)
    assert alignment.classify(record([], label="unknown")) is None


def test_product_and_question_leakage_are_rejected():
    def row(qid, asin, question):
        return {"id": str(qid), "asin": asin, "normalized_question_sha256": alignment.digest(question.encode()), "source_record_sha256": alignment.digest(str(qid).encode())}
    splits = {"train": [row(1, "P1", "question1")], "validation": [row(2, "P2", "question2")],
              "answerable": [row(3, "P3", "question3")], "unanswerable": [row(4, "P4", "question4")]}
    alignment.assert_disjoint(splits)
    splits["unanswerable"][0]["asin"] = "P1"
    with pytest.raises(ValueError, match="asin"):
        alignment.assert_disjoint(splits)
    splits["unanswerable"][0]["asin"] = "P4"
    splits["unanswerable"][0]["normalized_question_sha256"] = splits["train"][0]["normalized_question_sha256"]
    with pytest.raises(ValueError, match="normalized_question"):
        alignment.assert_disjoint(splits)
