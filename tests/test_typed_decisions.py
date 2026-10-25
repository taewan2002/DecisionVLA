import numpy as np
import pytest

from decisionvla.typed_decisions import DecisionSchema, action_decisions, read_decision, robot_schema


def question(kind, **kwargs):
    return {"type": kind, "instructions": "Choose based on the observation", **kwargs}


def test_typed_answers_order_identity_and_score():
    qs = [question("predicate", name="grip"),
          question("choice", choices=[{"value": True}, {"value": "true"}]),
          question("score", levels=[{"label": x} for x in ["low", "medium", "high"]])]
    schema = DecisionSchema.parse(qs)
    answers = schema.read([[.8, .2], [.2, .8], [.2, .5, .3]])
    assert answers[0] == {"type": "predicate", "name": "grip", "probability": .8}
    assert answers[1]["choice"] == "true"
    assert answers[1]["probabilities"][0]["value"] is True
    assert answers[2]["score"] == pytest.approx(1.1)
    assert answers[2]["probabilities"][0]["label"] == "low"
    assert read_decision(question("noul"), [.8, .2])["type"] == "predicate"
    assert DecisionSchema.parse(schema.as_list()) == schema


@pytest.mark.parametrize("raw", [
    {}, question("predicate", name=""), question("predicate", instructions=""),
    question("predicate", confidence=1), question("unknown"),
    question("choice", choices=[]), question("score", levels=[{"label": "one"}]),
    question("choice", choices=[{"value": 1}, {"value": 2}]),
    question("choice", choices=[{"value": "x"}, {"value": "x"}]),
    question("score", levels=[{"label": "a"}, {"label": "a"}]),
])
def test_reject_invalid_questions(raw):
    with pytest.raises(ValueError):
        DecisionSchema.parse([raw])


def test_reject_bad_probabilities_names_and_counts():
    q = question("predicate", name="same")
    for p in ([.9, .9], [float("nan"), .5], [-.1, 1.1], [.5, .5, 0]):
        with pytest.raises(ValueError):
            read_decision(q, p)
    for qs in ([], [q, q], [question("predicate")] * 57):
        with pytest.raises(ValueError):
            DecisionSchema.parse(qs)
    with pytest.raises(ValueError):
        DecisionSchema.parse([q]).read([])


def test_question_semantics_enter_prompt_and_hash():
    a = DecisionSchema.parse([question("score", levels=[{"label": "low"}, {"label": "high"}])])
    b = DecisionSchema.parse([question("score", instructions="Rate object height",
                                      levels=[{"label": "low"}, {"label": "high"}])])
    assert a.sha256 != b.sha256
    assert a.prompt("lift", [(4, 5)]) != b.prompt("lift", [(4, 5)])
    assert '"token_id":4' in a.prompt("lift", [(4, 5)])


def test_robot_profile_has_real_command_questions_and_no_progress_score():
    p = np.zeros((8, 7, 256))
    p[..., 0], p[..., -1] = .51, .49
    original = p.copy()
    result = action_decisions(p)
    np.testing.assert_array_equal(p, original)
    assert len(result["answers"]) == 56
    assert result["answers"][0]["choice"] == "0"
    assert result["answers"][6]["probability"] == .49
    assert result["calibrated"] is False
    assert {q.type for q in robot_schema().questions} == {"choice", "predicate"}
    assert result["answers"][0]["confidence"] > .8  # bimodality != correctness
