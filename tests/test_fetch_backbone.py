"""`scripts/fetch_backbone.py`·`robo_jev.model.backbone` 검사 — 가짜 Hub로 고정 revision·파일 해시·yaml 써 넣기·적재 전 대조를 본다 (Task 2b G0a).

`tests/test_fetch_tokenizer.py`와 같은 꼴이다. 네트워크·`transformers`·실제 가중치는 쓰지 않는다 — safetensors는 header
(8바이트 길이 + JSON)와 0 바이트로 만든 작은 파일이다.
"""

import copy
import functools
import importlib.util
import json
import struct
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml
from helpers import REPO

from robo_jev.model.backbone import (
    FILE_PATTERNS,
    MANIFEST_NAME,
    SMALL_FILES,
    config_structure,
    count_params,
    describe_backbone,
    fileset_digest,
    safetensors_header,
)

SCRIPT = REPO / "scripts" / "fetch_backbone.py"
CANDIDATES = REPO / "configs" / "model" / "candidates.yaml"
TWO_B = "Qwen/Qwen3.5-2B"
SHA_A = "1111111111111111111111111111111111111111"
SHA_B = "2222222222222222222222222222222222222222"


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("fetch_backbone", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tiny_safetensors(tensors: dict[str, list[int]]) -> bytes:
    """header + 0 바이트의 safetensors (BF16, 원소당 2바이트)."""
    header: dict[str, dict] = {}
    offset = 0
    for name, shape in tensors.items():
        size = 2
        for dim in shape:
            size *= dim
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    payload = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload + b"\0" * offset


def config_from_entry(entry: dict) -> dict:
    """yaml 항목의 구조 값에서 Hub의 config.json 꼴(text_config 안)을 되만든다 — 구조 대조가 통과하는 가짜."""
    counts = entry["layer_types"]
    types = []
    for index in range(sum(counts.values())):
        types.append("full_attention" if (index + 1) % entry["full_attention_interval"] == 0 else "linear_attention")
    linear = entry["linear_attention"]
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": entry["model_type"],
        "tie_word_embeddings": entry["tie_word_embeddings"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "layer_types": types,
            "full_attention_interval": entry["full_attention_interval"],
            "hidden_size": entry["hidden_size"],
            "intermediate_size": entry["intermediate_size"],
            "num_attention_heads": entry["num_attention_heads"],
            "num_key_value_heads": entry["num_key_value_heads"],
            "head_dim": entry["head_dim"],
            "linear_num_key_heads": linear["num_key_heads"],
            "linear_num_value_heads": linear["num_value_heads"],
            "linear_key_head_dim": linear["key_head_dim"],
            "linear_value_head_dim": linear["value_head_dim"],
            "linear_conv_kernel_dim": linear["conv_kernel_dim"],
            "vocab_size": entry["vocab_size"],
        },
        "vision_config": {"model_type": "qwen3_5"},
    }


TENSORS = {
    "model.language_model.embed_tokens.weight": [8, 4],
    "model.language_model.layers.0.mlp.up_proj.weight": [4, 4],
    "model.visual.patch_embed.proj.weight": [3, 2],
    "mtp.fc.weight": [2, 2],
}


class FakeHub:
    """id → {revision SHA → {파일명: bytes}}. `heads`는 revision을 주지 않았을 때의 현재 revision."""

    def __init__(self, repos: dict, heads: dict, licenses: dict | None = None) -> None:
        self.repos, self.heads, self.licenses = repos, heads, licenses or {}
        self.calls: list[tuple] = []

    def revision(self, identifier: str, revision: str | None) -> str:
        self.calls.append(("revision", identifier, revision))
        if identifier not in self.repos:
            raise LookupError(f"{identifier}: Hub에 없다")
        sha = self.heads[identifier] if revision is None else revision
        if sha not in self.repos[identifier]:
            raise LookupError(f"{identifier}@{revision}: 없는 revision")
        return sha

    def license(self, identifier: str, revision: str) -> str | None:
        self.calls.append(("license", identifier, revision))
        return self.licenses.get(identifier)

    def download(self, identifier: str, patterns: list[str], revision: str, target: Path) -> Path:
        self.calls.append(("download", identifier, tuple(patterns), revision))
        target.mkdir(parents=True, exist_ok=True)
        for name, payload in self.repos[identifier][revision].items():
            if any(fnmatch(name, pattern) for pattern in patterns):
                (target / name).write_bytes(payload)
        return target


@pytest.fixture()
def workspace(tmp_path):
    """임시 candidates.yaml(저장소의 2B 항목 그대로) + 그 항목과 맞는 가짜 Hub."""
    data = yaml.safe_load(CANDIDATES.read_text(encoding="utf-8"))
    entry = copy.deepcopy(next(item for item in data["candidates"] if item["id"] == TWO_B))
    entry["revision"] = SHA_A  # 저장소의 실제 SHA 대신 가짜 Hub의 revision을 고정한다
    config_path = tmp_path / "candidates.yaml"
    config_path.write_text("# 검사용 사본\n\n" + yaml.safe_dump({"version": data["version"], "candidates": [copy.deepcopy(entry)]}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    files_a = {
        "config.json": json.dumps(config_from_entry(entry)).encode("utf-8"),
        "model.safetensors-00001-of-00001.safetensors": tiny_safetensors(TENSORS),
        "model.safetensors.index.json": json.dumps({"weight_map": {name: "model.safetensors-00001-of-00001.safetensors" for name in TENSORS}}).encode("utf-8"),
        "tokenizer.json": b"{}",  # 받지 않는 파일
    }
    files_b = dict(files_a, **{"model.safetensors-00001-of-00001.safetensors": tiny_safetensors({**TENSORS, "model.language_model.extra": [2]})})
    hub = FakeHub({TWO_B: {SHA_A: files_a, SHA_B: files_b}}, heads={TWO_B: SHA_B}, licenses={TWO_B: "apache-2.0"})
    return {"config": config_path, "root": tmp_path / "models", "hub": hub, "entry": entry, "files_a": files_a}


def run(workspace, *extra) -> int:
    return script().main(["--root", str(workspace["root"]), "--config", str(workspace["config"]), *extra], hub=workspace["hub"])


def manifest_of(workspace) -> dict:
    return json.loads((workspace["root"] / MANIFEST_NAME).read_text(encoding="utf-8"))


def test_safetensors_header_and_param_counts_come_from_the_tensors_not_the_name(tmp_path):
    path = tmp_path / "w.safetensors"
    path.write_bytes(tiny_safetensors(TENSORS))
    assert set(safetensors_header(path)) == set(TENSORS)
    counts = count_params([path])
    assert counts == {"total": 32 + 16 + 6 + 4, "text": 48, "vision": 6, "mtp": 4, "other": 0, "tensors": 4}
    other = tmp_path / "dup.safetensors"
    other.write_bytes(tiny_safetensors({"mtp.fc.weight": [1]}))
    with pytest.raises(ValueError, match="겹친다"):
        count_params([path, other])


def test_config_structure_reads_text_config_and_falls_back_to_the_top_level_tie_flag():
    data = yaml.safe_load(CANDIDATES.read_text(encoding="utf-8"))
    for entry in data["candidates"]:
        structure = config_structure(config_from_entry(entry))
        assert structure == {key: entry[key] for key in structure}
    nine = next(item for item in data["candidates"] if item["id"] == "Qwen/Qwen3.5-9B")
    config = config_from_entry(nine)
    config["tie_word_embeddings"] = False
    assert config_structure(config)["tie_word_embeddings"] is False
    config["text_config"]["tie_word_embeddings"] = True
    assert config_structure(config)["tie_word_embeddings"] is True


def test_default_fetch_takes_the_yaml_revision_hashes_every_file_and_writes_manifest_and_yaml(workspace, capsys):
    assert run(workspace, "--id", TWO_B) == 0
    hub = workspace["hub"]
    assert ("revision", TWO_B, workspace["entry"]["revision"]) in hub.calls  # yaml의 40-hex revision을 고정으로 쓴다
    assert ("download", TWO_B, tuple(FILE_PATTERNS), workspace["entry"]["revision"]) in hub.calls
    target = workspace["root"] / TWO_B
    assert not (target / "tokenizer.json").exists()  # 패턴 밖의 파일은 받지 않는다

    manifest = manifest_of(workspace)
    entry = manifest["models"][TWO_B]
    assert entry["revision"] == workspace["entry"]["revision"] and entry["license"] == "apache-2.0" and entry["fetched_at"]
    assert set(entry["files"]) == {"config.json", "model.safetensors-00001-of-00001.safetensors", "model.safetensors.index.json"}
    for name, spec in entry["files"].items():
        assert spec["bytes"] == (target / name).stat().st_size and len(spec["sha256"]) == 64
    assert entry["digest"] == fileset_digest(entry["files"]) and entry["bytes_total"] == sum(s["bytes"] for s in entry["files"].values())
    assert entry["params"] == {"total": 58, "text": 48, "vision": 6, "mtp": 4, "other": 0, "tensors": 4}
    assert entry["pinned"] == {"revision": workspace["entry"]["revision"], "sha256": None}
    assert manifest["root"] == str(workspace["root"].resolve())

    updated = yaml.safe_load(workspace["config"].read_text(encoding="utf-8"))["candidates"][0]
    assert updated["revision"] == entry["revision"] and updated["license"] == "apache-2.0"
    assert updated["params_total"] == 48 and updated["params_checkpoint"] == 58
    assert {key: updated[key] for key in workspace["entry"] if key not in ("params_total", "params_checkpoint")} == {
        key: workspace["entry"][key] for key in workspace["entry"] if key not in ("params_total", "params_checkpoint")
    }
    assert workspace["config"].read_text(encoding="utf-8").startswith("# 검사용 사본\n")  # 머리말은 남는다

    # 받은 파일은 곧바로 manifest와 대조되어 읽힌다
    described = describe_backbone(TWO_B, root=workspace["root"])
    assert described["revision"] == entry["revision"] and described["verified"] == "sizes+small-files"
    assert describe_backbone(TWO_B, root=workspace["root"], full=True)["verified"] == "full"
    out = capsys.readouterr().out
    assert TWO_B in out and entry["revision"] in out and entry["digest"] in out


def test_loading_rejects_missing_wrong_sized_or_altered_files_and_names_the_fetch_script(workspace):
    assert run(workspace, "--id", TWO_B) == 0
    root = workspace["root"]
    target = root / TWO_B
    with pytest.raises(FileNotFoundError, match="fetch_backbone.py"):
        describe_backbone("Qwen/Qwen3.5-4B", root=root)

    config = target / "config.json"
    original = config.read_bytes()
    config.write_bytes(original.replace(b"qwen3_5", b"qwen3_6", 1))  # 같은 크기, 다른 내용 — 작은 파일은 언제나 해시를 본다
    with pytest.raises(ValueError, match="sha256이 다르다") as excinfo:
        describe_backbone(TWO_B, root=root)
    assert "--from-manifest" in str(excinfo.value)
    config.write_bytes(original)

    weights = target / "model.safetensors-00001-of-00001.safetensors"
    payload = weights.read_bytes()
    weights.write_bytes(payload[:-1] + b"\1")  # 같은 크기, 다른 내용 — safetensors는 full일 때만 잡는다
    describe_backbone(TWO_B, root=root)
    with pytest.raises(ValueError, match="sha256이 다르다"):
        describe_backbone(TWO_B, root=root, full=True)
    weights.write_bytes(payload + b"\0")
    with pytest.raises(ValueError, match="크기가 다르다"):
        describe_backbone(TWO_B, root=root)
    weights.unlink()
    with pytest.raises(ValueError, match="파일이 없다"):
        describe_backbone(TWO_B, root=root)


def test_from_manifest_reproduces_the_same_files_and_a_changed_file_fails_and_leaves_nothing(workspace, tmp_path):
    assert run(workspace, "--id", TWO_B) == 0
    manifest = manifest_of(workspace)
    handed_over = tmp_path / "handoff-manifest.json"
    handed_over.write_text(json.dumps(manifest), encoding="utf-8")
    other = dict(workspace, root=tmp_path / "models-2")
    other["hub"].heads[TWO_B] = SHA_B  # 현재 revision이 달라도 manifest의 revision을 받는다
    assert run(other, "--from-manifest", str(handed_over)) == 0
    assert manifest_of(other)["models"][TWO_B]["digest"] == manifest["models"][TWO_B]["digest"]
    assert ("download", TWO_B, tuple(FILE_PATTERNS), SHA_A) in other["hub"].calls

    # 그 revision의 파일이 바뀌면 실패하고 받은 파일을 남기지 않으며 manifest는 그대로다
    third = dict(workspace, root=tmp_path / "models-3")
    third["hub"].repos[TWO_B][SHA_A]["config.json"] = b'{"model_type": "qwen3_5"}'
    assert run(third, "--from-manifest", str(handed_over)) == 1
    assert not (third["root"] / MANIFEST_NAME).exists()
    assert not any((third["root"] / TWO_B).glob("*.safetensors")) and not (third["root"] / TWO_B / "config.json").exists()

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"id": TWO_B}), encoding="utf-8")
    with pytest.raises(SystemExit):
        run(workspace, "--from-manifest", str(incomplete))


def test_expected_digest_pins_the_file_set_and_a_structure_change_is_refused(workspace, tmp_path, capsys):
    hub = workspace["hub"]
    assert run(workspace, "--id", TWO_B) == 0
    digest = manifest_of(workspace)["models"][TWO_B]["digest"]
    # 지문이 맞으면(대소문자 무관) 성공, 다른 revision의 묶음이면 실패하고 파일을 지운다
    assert run(workspace, "--id", TWO_B, "--expect-sha256", digest.upper()) == 0
    assert manifest_of(workspace)["models"][TWO_B]["pinned"] == {"revision": SHA_A, "sha256": digest}
    assert run(workspace, "--id", TWO_B, "--revision", SHA_B, "--expect-sha256", digest) == 1
    err = capsys.readouterr().err
    assert "HashMismatch" in err and "지문" in err
    assert not (workspace["root"] / TWO_B / "config.json").exists()
    assert manifest_of(workspace)["models"][TWO_B]["revision"] == SHA_A  # 실패한 시도는 manifest를 바꾸지 않는다

    # 받은 config.json의 구조가 yaml과 다르면 받아도 manifest·yaml에 적지 않는다
    wrong = json.loads(hub.repos[TWO_B][SHA_A]["config.json"])
    wrong["text_config"]["hidden_size"] = 4096
    hub.repos[TWO_B][SHA_A]["config.json"] = json.dumps(wrong).encode("utf-8")
    before = manifest_of(workspace)
    assert run(workspace, "--id", TWO_B) == 1
    assert "StructureMismatch" in capsys.readouterr().err and manifest_of(workspace) == before

    # yaml에 없는 id는 받지 않는다
    assert run(workspace, "--id", "Qwen/Qwen3.5-4B") == 1
    assert "candidates.yaml" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run(workspace)


def test_yaml_edited_while_the_weights_download_is_not_clobbered_by_the_write_back(workspace):
    """받는 데 수십 분이 걸린다 — 그 사이에 고친 yaml(머리말·다른 항목)이 쓰기 직전에 다시 읽혀 남아야 한다."""
    hub = workspace["hub"]
    config_path = workspace["config"]
    original_download = hub.download

    def download_and_edit(identifier, patterns, revision, target):
        path = original_download(identifier, patterns, revision, target)
        text = config_path.read_text(encoding="utf-8")
        config_path.write_text("# 받는 동안 고친 머리말\n" + text.replace("role: main", "role: separate"), encoding="utf-8")
        return path

    hub.download = download_and_edit
    assert run(workspace, "--id", TWO_B) == 0
    text = config_path.read_text(encoding="utf-8")
    assert text.startswith("# 받는 동안 고친 머리말\n# 검사용 사본\n")
    updated = yaml.safe_load(text)["candidates"][0]
    assert updated["role"] == "separate" and updated["params_total"] == 48 and updated["revision"] == SHA_A


def test_small_files_are_the_ones_always_hashed():
    assert set(SMALL_FILES) <= {"config.json", "model.safetensors.index.json", "generation_config.json"}
    assert all(any(fnmatch(name, pattern) for pattern in FILE_PATTERNS) for name in SMALL_FILES)
