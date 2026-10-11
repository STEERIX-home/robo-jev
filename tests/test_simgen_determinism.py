"""결정성: 같은 seed → 같은 JSONL 바이트 (Task v0a 지시서 §3.5·§5 '결정성', docs/12 §12 '재현').

레코드는 (생성기 버전, 속성 목록 버전, seed, 강제한 분야, 쌍 변형)으로 다시 만들 수 있다 — `generator` 칸이 그 열쇠를 다 담는다.
"""

from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.generate import generate_batch, generate_record, plan_units
from robo_jev.simgen.record import dumps
from robo_jev.simgen.spec import stream


def _jsonl(n: int, seed: int) -> bytes:
    return "".join(dumps(r) + "\n" for r in generate_batch(n, base_seed=seed, cfg=default_config(), git="fixed")).encode("utf-8")


def test_same_seed_gives_byte_identical_jsonl():
    assert _jsonl(60, 3) == _jsonl(60, 3)


def test_different_seed_gives_different_records():
    assert _jsonl(20, 3) != _jsonl(20, 4)


def test_each_record_is_reproducible_from_its_generator_block():
    for record in generate_batch(40, base_seed=9, cfg=default_config(), git="fixed"):
        gen = record["generator"]
        index = int(record["id"].split("-")[-1])
        again = generate_record(gen["seed"], index, gen["forced"]["domain"], variant=gen["variant"], pair=record["meta"]["pair"], git="fixed")
        assert dumps(again) == dumps(record), record["id"]


def test_named_streams_are_independent_of_each_other():
    """갈래 하나를 더 써도 다른 갈래의 수가 바뀌지 않는다(이름으로 나눈 SeedSequence)."""
    a = stream(5, "world").random(3)
    _ = stream(5, "observe").random(100)
    b = stream(5, "world").random(3)
    assert (a == b).all()
    assert not (stream(5, "world").random(3) == stream(5, "skin", 0).random(3)).all()


def test_unit_plan_is_deterministic():
    assert plan_units(500, 1) == plan_units(500, 1)
