"""Hugging Face Hub side of the export (Task hf0 §5): the token file, **private** repos, upload, the release tag, the remote listing.

**Token.** :func:`load_token` reads ``HF_TOKEN=…`` from ``~/.config/robo-jev/huggingface.env`` only — not from the environment or the
command line — and refuses a file that the group or others can read (like the OpenRouter and RunPod keys). The value never goes to a
log, an exception or a file: callers pass it straight to :class:`huggingface_hub.HfApi` and :func:`redact` scrubs error texts.

**Private only.** :func:`ensure_private_repo` creates the repo with ``private=True`` (or finds it) and then **checks** that the Hub says
it is private before anything is uploaded — an existing public repo of the same name stops the upload. Nothing here can make a repo
public or delete one.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_TOKEN_FILE", "UnsafeTokenFile", "ensure_private_repo", "load_token", "redact", "remote_listing", "tag_release", "upload"]

DEFAULT_TOKEN_FILE = Path.home() / ".config" / "robo-jev" / "huggingface.env"
_TOKEN_NAME = "HF_TOKEN"
_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{8,}")


class UnsafeTokenFile(PermissionError):
    """The token file can be read by the group or others."""


def redact(text: Any) -> str:
    """Error text without anything that looks like a Hugging Face token."""
    return _TOKEN_PATTERN.sub("hf_***", str(text))


def load_token(path: str | Path = DEFAULT_TOKEN_FILE) -> str:
    """``HF_TOKEN`` from the env file (``KEY=value`` lines, ``#`` comments). Refuses group/other-readable files."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no Hugging Face token file at {path}")
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise UnsafeTokenFile(f"{path} is readable by group/others (mode {oct(mode & 0o777)}) — chmod 600 it")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        if name == _TOKEN_NAME:
            value = value.strip().strip("'\"")
            if not value:
                break
            return value
    raise ValueError(f"{path}: no {_TOKEN_NAME}=… line")


def ensure_private_repo(api: Any, repo_id: str, *, allow_existing_public: bool = False) -> dict[str, Any]:
    """Create `repo_id` as a private model repo (or find it) and confirm it is private; raise otherwise. Returns ``{repo_id, private,
    created}``.

    ``allow_existing_public`` accepts a repo that ALREADY exists and that its owner has made public (the owner's decision, taken on the
    Hub; user 2026-10-16: "A로 진행해, 업로드 도구 고쳐"): the upload then goes to that public repo. A missing repo is still created
    private, and the visibility is never changed here."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = api.model_info(repo_id)
        created = False
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        info = api.model_info(repo_id)
        created = True
    if info.private is not True:
        if allow_existing_public and not created:
            return {"repo_id": repo_id, "private": False, "created": False}
        raise PermissionError(f"{repo_id} exists and is not private — refusing to upload (making repos public is the owner's decision)")
    return {"repo_id": repo_id, "private": True, "created": created}


def upload(api: Any, repo_id: str, folder: str | Path, *, message: str) -> dict[str, Any]:
    """Upload `folder` to the (already checked private) repo in one commit."""
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder), commit_message=message)
    return {"commit": getattr(commit, "oid", None), "url": getattr(commit, "commit_url", None)}


def _resolve(api: Any, repo_id: str, revision: str) -> str:
    """The commit a branch or tag points to (an annotated tag's ``target_commit`` in ``list_repo_refs`` is the tag object, not the commit)."""
    return api.model_info(repo_id, revision=revision).sha


def tag_release(api: Any, repo_id: str, tag: str, *, message: str, revision: str | None = None) -> dict[str, Any]:
    """Create the release tag on `revision` (default: the head of main); an existing tag is left as it is. Returns the tag, the commit it
    resolves to, the head of main, and whether it was created now."""
    refs = api.list_repo_refs(repo_id, repo_type="model")
    created = tag not in {ref.name for ref in refs.tags}
    if created:
        api.create_tag(repo_id, tag=tag, tag_message=message, revision=revision, repo_type="model")
    return {"tag": tag, "commit": _resolve(api, repo_id, tag), "main": _resolve(api, repo_id, "main"), "created": created}


def remote_listing(api: Any, repo_id: str, *, revision: str | None = None) -> dict[str, Any]:
    """What the Hub holds: privacy, files with sizes (and LFS sha256), tags, head commit."""
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    files = []
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files.append({"path": sibling.rfilename, "bytes": sibling.size, "lfs_sha256": sha})
    refs = api.list_repo_refs(repo_id, repo_type="model")
    return {
        "repo_id": repo_id, "private": info.private, "sha": info.sha, "files": files, "bytes": sum(f["bytes"] or 0 for f in files),
        "tags": {ref.name: _resolve(api, repo_id, ref.name) for ref in refs.tags},
        "card": info.card_data.to_dict() if info.card_data is not None else None, "url": f"https://huggingface.co/{repo_id}",
    }
