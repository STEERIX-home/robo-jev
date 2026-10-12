"""Hugging Face export (Task hf0 §1, §4, §5) — on the small random Qwen (CPU) with a fake LoRA checkpoint in the training format.

* The PEFT ``LoraConfig`` equals the training LoRA; the checkpoint → PEFT key map is explicit and fails loudly on missing, unexpected,
  malformed or misshapen keys; the saved adapter holds the checkpoint's tensors bit for bit under PEFT's names.
* End to end: export → ``RoboJev.from_pretrained`` of the adapter folder gives byte-identical answers to the training path
  (``attach_lora`` + the checkpoint values, as ``diagnose_decisions.py --checkpoint`` loads it); the merged folder gives the same answers
  up to float32 rounding. No pickle files; ``robo_jev.json`` carries the prompt, codes, tokenizer and provenance.
* Model cards: metadata, no unfilled placeholders, the organisation repo ids. Hub helpers: the token file, private-only repos.
"""

import functools
import json
import stat

import pytest
import torch
import yaml
from helpers import FIXTURES, REPO

pytest.importorskip("transformers")
pytest.importorskip("peft")
from robo_jev.checkpoint import CHECKPOINT_FORMAT  # noqa: E402
from robo_jev.decisions.prompt import PROMPT_VERSION, fixed_text_digest  # noqa: E402
from robo_jev.decisions.records import read_records, shuffle_key  # noqa: E402
from robo_jev.decisions.train import DEFAULT_LORA, _load_model_state  # noqa: E402
from robo_jev.hf import export as E  # noqa: E402
from robo_jev.hf import hub as H  # noqa: E402
from robo_jev.infer import RoboJev, file_sha256  # noqa: E402
from robo_jev.model.backbone_qwen import QwenBackbone, torch_reference_kernels  # noqa: E402
from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer  # noqa: E402
from robo_jev.model.zero_shot import verify_codes  # noqa: E402
from robo_jev.train import attach_lora  # noqa: E402

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
TINY_LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "targets": list(DEFAULT_LORA["targets"])}


@functools.lru_cache(maxsize=1)
def tokenizer():
    return load_tokenizer(available_tokenizer()[0])


def tiny_backbone():
    return QwenBackbone.tiny(seed=4, vocab_size=tokenizer().get_vocab_size())


def fake_state(lora=TINY_LORA, *, seed=0):
    """A ``robo-jev-checkpoint-v0`` LoRA checkpoint for the tiny model: random bf16 adapter values under the training keys."""
    backbone = tiny_backbone()
    attach_lora(backbone, lora)
    generator = torch.Generator().manual_seed(seed)
    saved = {name: (torch.randn(p.shape, generator=generator) * 0.05).to(torch.bfloat16)
             for name, p in backbone.text.named_parameters() if "lora_" in name}
    manifest = {
        "kind": "decisions-train-v0",
        "model": {"id": E.BASE_ID, "mode": "lora", "lora": dict(lora), "lora_dtype": "bfloat16", "trainable_parameters": sum(t.numel() for t in saved.values()),
                  "base": {"model_id": E.BASE_ID, "digest": "tiny", "revision": E.BASE_REVISION, "dtype": "float32", "vocab_size": tokenizer().get_vocab_size()}},
        "tokenizer": {"id": available_tokenizer()[0], "sha256": file_sha256(available_tokenizer()[1])},
        "prompt": {"version": PROMPT_VERSION, "digest": fixed_text_digest(PROMPT_VERSION), "format": "chat", "mode": "direct"},
        "objective": {"main": "kl"}, "selection": {"evaluated": "final checkpoint (pre-registered)"},
        "data": {"records": 2, "dev_ids": ["a", "b"], "records_sha256": "0" * 64}, "git": None,
    }
    return {"format": CHECKPOINT_FORMAT, "run_id": "tiny", "step": 7, "status": "completed", "model": saved, "optimizer": {}, "scheduler": {},
            "rng": {}, "sampler": {}, "progress": None, "config": {"mode": "lora", "lr": 2e-4, "epochs": 1}, "manifest": manifest}


def test_the_peft_config_equals_the_training_lora():
    shipped = yaml.safe_load((REPO / "configs/train/decisions-v0-lora-h100.yaml").read_text(encoding="utf-8"))["lora"]
    assert shipped == {**DEFAULT_LORA, "targets": list(DEFAULT_LORA["targets"])}
    config = E.lora_config(shipped)
    assert (config.r, config.lora_alpha, config.lora_dropout, config.bias) == (32, 64, 0.0, "none")
    assert set(config.target_modules) == set(shipped["targets"]) and str(config.task_type) in ("CAUSAL_LM", "TaskType.CAUSAL_LM")
    assert (config.base_model_name_or_path, config.revision) == ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc")
    assert not config.use_rslora and not config.use_dora


def test_the_key_map_is_explicit_and_fails_loudly():
    state = fake_state()
    saved = state["model"]
    assert E.adapter_key_map(["layers.0.linear_attn.out_proj.lora_A.default.weight"]) == {
        "layers.0.linear_attn.out_proj.lora_A.default.weight": "base_model.model.model.layers.0.linear_attn.out_proj.lora_A.default.weight"}
    with pytest.raises(ValueError, match="unexpected checkpoint key"):
        E.adapter_key_map(["model.layers.0.linear_attn.out_proj.lora_A.weight"])
    model = E.build_peft_model(tiny_backbone().model, TINY_LORA)
    assert len(E.load_adapter_state(model, saved)) == len(saved)
    first = next(iter(saved))
    with pytest.raises(ValueError, match="missing 1"):
        E.load_adapter_state(model, {k: v for k, v in saved.items() if k != first})
    with pytest.raises(ValueError, match="unexpected 1"):
        E.load_adapter_state(model, {**saved, "layers.99.mlp.up_proj.lora_A.default.weight": saved[first]})
    with pytest.raises(ValueError, match="shape"):
        E.load_adapter_state(model, {**saved, first: torch.zeros(3, 3, dtype=torch.bfloat16)})


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export the fake checkpoint (adapter + merged, merged kept in float32 for a tight comparison) next to a saved tiny base."""
    root = tmp_path_factory.mktemp("hf")
    state = fake_state()
    checkpoint = root / "checkpoint.pt"
    torch.save(state, checkpoint)
    base_dir = root / "base"
    tiny_backbone().model.save_pretrained(base_dir)
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    loaded = E.read_training_checkpoint(checkpoint)
    from transformers import AutoModelForCausalLM

    peft_model = E.build_peft_model(AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32), TINY_LORA)
    E.load_adapter_state(peft_model, loaded["model"])
    folders = {"adapter": root / "adapter", "merged": root / "merged"}
    weights = {"adapter": E.save_adapter(peft_model, loaded["model"], folders["adapter"]),
               "merged": E.save_merged(peft_model, folders["merged"], dtype=torch.float32)}
    for variant, folder in folders.items():
        metadata = E.build_metadata(loaded, variant=variant, checkpoint_path=checkpoint, checkpoint_sha256=file_sha256(checkpoint), tokenizer=tokenizer(),
                                    tokenizer_sha256=file_sha256(available_tokenizer()[1]), repo=REPO, weights=weights[variant])
        E.write_common_files(folder, metadata=metadata, tokenizer_json=available_tokenizer()[1], tokenizer_config=root / "tokenizer_config.json",
                             chat_template=root / "chat_template.jinja", card=E.render_card(variant))
    return {"root": root, "state": state, "base": base_dir, "folders": folders, "weights": weights}


def test_the_export_writes_safetensors_and_metadata_only(exported):
    adapter, merged = exported["folders"]["adapter"], exported["folders"]["merged"]
    common = {"README.md", "chat_template.jinja", "robo_jev.json", "robo_jev_decide.py", "tokenizer.json", "tokenizer_config.json"}
    assert set(E.check_no_pickle(adapter)) == common | {"adapter_config.json", "adapter_model.safetensors"}
    assert set(E.check_no_pickle(merged)) == common | {"config.json", "generation_config.json", "model.safetensors"}
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert (config["base_model_name_or_path"], config["revision"], config["task_type"], config["peft_type"]) == (E.BASE_ID, E.BASE_REVISION, "CAUSAL_LM", "LORA")
    assert (config["r"], config["lora_alpha"], config["bias"], set(config["target_modules"])) == (4, 8, "none", set(TINY_LORA["targets"]))
    assert json.loads((merged / "config.json").read_text(encoding="utf-8"))["architectures"] == ["Qwen3_5ForCausalLM"]
    assert exported["weights"]["adapter"]["dtype"] == "bfloat16" and exported["weights"]["adapter"]["tensors"] == len(exported["state"]["model"])
    meta = json.loads((merged / "robo_jev.json").read_text(encoding="utf-8"))
    assert meta["prompt"]["version"] == PROMPT_VERSION and meta["prompt"]["digest"] == fixed_text_digest(PROMPT_VERSION)
    codes, ids = verify_codes(tokenizer(), 11)
    assert meta["answer_codes"]["token_ids"] == dict(zip(codes, ids)) and meta["score_anchors"] == [i / 10 for i in range(11)]
    assert meta["tokenizer"]["sha256"] == file_sha256(merged / "tokenizer.json") == exported["state"]["manifest"]["tokenizer"]["sha256"]
    assert meta["source"]["checkpoint_sha256"] == file_sha256(exported["root"] / "checkpoint.pt") and meta["base"]["revision"] == E.BASE_REVISION
    assert "dev_ids" not in meta["training"]["data"] and len(meta["training"]["data"]["dev_ids_sha256"]) == 64
    assert meta["repos"] == {"merged": "Steerix-org/robo-jev-2b", "adapter": "Steerix-org/robo-jev-2b-lora", "code": E.GITHUB}
    assert (merged / "robo_jev_decide.py").read_bytes() == E.STANDALONE.read_bytes()


def test_check_no_pickle_refuses_pickles(tmp_path):
    (tmp_path / "ok.json").write_text("{}", encoding="utf-8")
    assert E.check_no_pickle(tmp_path) == ["ok.json"]
    (tmp_path / "weights.bin").write_bytes(b"{}")
    with pytest.raises(ValueError, match="pickle-type"):
        E.check_no_pickle(tmp_path)
    (tmp_path / "weights.bin").unlink()
    (tmp_path / "hidden.json").write_bytes(b"\x80\x04K.")
    with pytest.raises(ValueError, match="pickle or zip"):
        E.check_no_pickle(tmp_path)
    (tmp_path / "hidden.json").write_bytes(b"PK\x03\x04")
    with pytest.raises(ValueError, match="pickle or zip"):
        E.check_no_pickle(tmp_path)


def _original_path(state):
    """What ``diagnose_decisions.py --checkpoint`` does for a LoRA run: the base backbone + ``attach_lora`` + the checkpoint values."""
    backbone = tiny_backbone()
    attach_lora(backbone, state["manifest"]["model"]["lora"])
    _load_model_state(backbone, "lora", state["model"])
    return RoboJev(backbone.model, tokenizer())


def test_the_exported_folders_answer_like_the_training_checkpoint(exported):
    with torch_reference_kernels():
        original = _original_path(exported["state"])
        adapter = RoboJev.from_pretrained(exported["folders"]["adapter"], device="cpu", dtype=torch.float32, base_model=exported["base"])
        merged = RoboJev.from_pretrained(exported["folders"]["merged"], device="cpu", dtype=torch.float32)
        base_only = RoboJev(tiny_backbone().model, tokenizer())
        moved = 0.0
        for record in read_records(REQUESTS)[:4]:
            request = record["request"]
            kwargs = {"orders": "balanced", "key": shuffle_key(record), "seed": 1}
            a = original.score_request(request["state"], request["questions"], **kwargs)
            b = adapter.score_request(request["state"], request["questions"], **kwargs)
            c = merged.score_request(request["state"], request["questions"], **kwargs)
            d = base_only.score_request(request["state"], request["questions"], **kwargs)
            for name in a:
                probs = [l["code_probs"] for l in a[name]["listings"]]
                assert probs == [l["code_probs"] for l in b[name]["listings"]]  # byte-identical
                for x, y in zip(probs, [l["code_probs"] for l in c[name]["listings"]]):
                    assert max(abs(u - v) for u, v in zip(x, y)) < 1e-4
                for x, y in zip(probs, [l["code_probs"] for l in d[name]["listings"]]):
                    moved = max(moved, max(abs(u - v) for u, v in zip(x, y)))
        assert moved > 1e-3  # the adapter does change the answers (the comparison is not vacuous)


def test_the_bf16_merged_folder_loads(tmp_path):
    state = fake_state(seed=1)
    peft_model = E.build_peft_model(tiny_backbone().model, TINY_LORA)
    E.load_adapter_state(peft_model, state["model"])
    info = E.save_merged(peft_model, tmp_path / "merged")
    assert info["merge_dtype"] == "float32" and info["dtype"] == "bfloat16" and info["files"] == ["model.safetensors"]
    from safetensors import safe_open

    with safe_open(str(tmp_path / "merged" / "model.safetensors"), framework="pt") as handle:
        assert {handle.get_tensor(k).dtype for k in handle.keys()} == {torch.bfloat16}
        assert not any("lora_" in k for k in handle.keys())


def test_from_pretrained_refuses_a_folder_with_another_prompt_or_tokenizer(exported, tmp_path):
    import shutil

    folder = tmp_path / "merged"
    shutil.copytree(exported["folders"]["merged"], folder)
    meta = json.loads((folder / "robo_jev.json").read_text(encoding="utf-8"))
    (folder / "robo_jev.json").write_text(json.dumps({**meta, "prompt": {**meta["prompt"], "version": "v0c-p2"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="expects prompt"):
        RoboJev.from_pretrained(folder, device="cpu", dtype=torch.float32)
    (folder / "robo_jev.json").write_text(json.dumps({**meta, "tokenizer": {**meta["tokenizer"], "sha256": "0" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="training tokenizer"):
        RoboJev.from_pretrained(folder, device="cpu", dtype=torch.float32)
    (folder / "robo_jev.json").unlink()
    with pytest.raises(FileNotFoundError, match="not a robo-jev export"):
        RoboJev.from_pretrained(folder, device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("variant", ["merged", "adapter"])
def test_the_model_cards(variant):
    card = E.render_card(variant, example_output="Example output.")
    assert card.startswith("---\n") and "{{" not in card and "Example output." in card
    meta = yaml.safe_load(card.split("---\n")[1])
    assert meta["license"] == "apache-2.0" and meta["base_model"] == "Qwen/Qwen3.5-2B" and meta["language"] == ["en", "ko"]
    assert (meta["base_model_relation"], meta["library_name"]) == (("finetune", "transformers") if variant == "merged" else ("adapter", "peft"))
    assert meta["pipeline_tag"] == "robotics" and meta["inference"] is False
    assert {"robotics", "decision-making", "physical-reasoning", "calibration", "lora"} <= set(meta["tags"])
    repo = E.MERGED_REPO if variant == "merged" else E.ADAPTER_REPO
    assert f'load("{repo}")' in card and f"--model {repo}" in card and "Steerix-admin" not in card
    assert "figures/bench-v0-physics-light.png" in card and "figures/bench-v0-physics-dark.png" in card


def test_repo_ids_point_to_the_organisation():
    assert (E.MERGED_REPO, E.ADAPTER_REPO, E.TAG) == ("Steerix-org/robo-jev-2b", "Steerix-org/robo-jev-2b-lora", "v0.1")
    for path in (E.STANDALONE, REPO / "src/robo_jev/infer.py", REPO / "scripts/export_hf.py", E.CARD_TEMPLATE):
        assert "Steerix-admin" not in path.read_text(encoding="utf-8")


def test_the_token_file_must_be_private_and_is_never_echoed(tmp_path):
    secret = "hf_" + "A1b2C3d4E5f6G7h8I9j0"
    path = tmp_path / "huggingface.env"
    path.write_text(f"# token\nexport HF_TOKEN='{secret}'\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(H.UnsafeTokenFile):
        H.load_token(path)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert H.load_token(path) == secret
    assert secret not in H.redact(f"401 for token {secret}") and "hf_***" in H.redact(secret)
    path.write_text("OTHER=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no HF_TOKEN"):
        H.load_token(path)


class _FakeInfo:
    def __init__(self, private):
        self.private = private


class _FakeApi:
    def __init__(self, existing=None):
        self.existing = dict(existing or {})
        self.created = []

    def model_info(self, repo_id, **_):
        import httpx
        from huggingface_hub.utils import RepositoryNotFoundError

        if repo_id not in self.existing:
            raise RepositoryNotFoundError("missing", response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/api/models/x")))
        return _FakeInfo(self.existing[repo_id])

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        assert repo_type == "model" and private is True and exist_ok is True
        self.created.append(repo_id)
        self.existing[repo_id] = private


def test_repos_are_created_private_and_a_public_repo_stops_the_upload():
    api = _FakeApi()
    assert H.ensure_private_repo(api, "Steerix-org/robo-jev-2b") == {"repo_id": "Steerix-org/robo-jev-2b", "private": True, "created": True}
    assert api.created == ["Steerix-org/robo-jev-2b"]
    assert H.ensure_private_repo(api, "Steerix-org/robo-jev-2b")["created"] is False
    with pytest.raises(PermissionError, match="not private"):
        H.ensure_private_repo(_FakeApi({"Steerix-org/robo-jev-2b": False}), "Steerix-org/robo-jev-2b")
