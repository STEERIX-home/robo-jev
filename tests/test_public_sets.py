"""공개 세트 변환 (Task D1 D3, 선택): 합성 행 → 계약을 지나는 레코드, 합의 없는 행 제외, 평가 전용 manifest."""

import json

from robo_jev.contracts import model_input, validate_record
from robo_jev.data.public_sets import PUBLIC_SETS_VERSION, boolq_record, convert, multinli_record, write_public_set


def test_public_rows_become_valid_state_first_records_and_an_evaluation_only_manifest(tmp_path):
    boolq = [{"question": "is the sky blue", "passage": "The sky appears blue in daylight.", "answer": True}, {"question": "is water dry", "passage": "Water is wet.", "answer": False}]
    records, summary = convert("boolq", boolq)
    assert summary == {"rows": 2, "records": 2, "skipped_no_gold": 0, "answers": {"False": 1, "True": 1}}
    assert records[0]["request"]["questions"][0]["type"] == "boolean" and records[0]["labels"][0]["answer"] is True
    assert model_input(records[0])["request"]["state"] == {"passage": "The sky appears blue in daylight.", "question": "is the sky blue"}
    nli = [
        {"sentence1": "A man eats.", "sentence2": "Someone is eating.", "gold_label": "entailment", "genre": "fiction", "pairID": "1"},
        {"sentence1": "A man eats.", "sentence2": "Nobody eats.", "gold_label": "contradiction", "genre": "fiction", "pairID": "2"},
        {"sentence1": "A man eats.", "sentence2": "He likes pasta.", "gold_label": "-", "genre": "fiction", "pairID": "3"},
    ]
    records, summary = convert("multinli", nli)
    assert summary["records"] == 2 and summary["skipped_no_gold"] == 1 and summary["answers"] == {"c_contradiction": 1, "c_entailment": 1}
    assert [c["id"] for c in records[0]["request"]["questions"][0]["criteria"]] == ["c_entailment", "c_neutral", "c_contradiction"]
    assert multinli_record(9, nli[2]) is None and boolq_record(3, boolq[1])["origin_group"] == "public/boolq/00003"
    for record in records:
        validate_record(record)
        assert record["split"] == "dev" and record["provenance"]["public_source"] == "multinli"
    manifest = write_public_set(tmp_path, "multinli", records, summary, source={"url": "u", "license": "l", "split": "dev_matched"}, source_file=None)
    assert manifest["trainer_manifest"] is False and manifest["purpose"] == "evaluation-only" and manifest["generator"] == PUBLIC_SETS_VERSION
    written = [json.loads(line) for line in (tmp_path / "multinli" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert written == records and manifest["files"]["records.jsonl"]["records"] == 2
