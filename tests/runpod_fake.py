"""가짜 RunPod API (REST v2) — 검사가 진짜 클라이언트 코드(`RunPodClient`)를 네트워크 없이 돌리게 한다 (Task cloud1).

전송 함수 하나(:meth:`FakeRunPodApi.__call__`)로 `GET /catalog/gpus`·`GET|POST /pods`·`GET|DELETE /pods/{id}`·`GET /network-volumes`·
`GET /billing/pods`를 메모리에서 흉내 낸다. 모든 요청을 (method, path, body, headers)로 **기록**한다 — 만들기·상태·지우기가 몇 번 불렸는지,
어떤 본문이 갔는지, 키가 어디에 실렸는지를 검사가 본다. 실패는 손잡이로 고른다:

* ``create_mode``: ``"ok"`` · ``"timeout_after_create"``(pod는 생겼는데 응답이 끊긴다 — 애매한 POST) · ``"timeout_before_create"`` ·
  ``"http_500"`` · ``"http_409"``(재고 없음)
* ``delete_failures``: 처음 N번의 DELETE가 500 — 그 뒤에 지워진다 · ``delete_never``: 지우기가 끝내 되지 않는다
* ``cost_override``: 만든 pod의 `cost`를 카탈로그와 다르게 (만든 뒤 단가 불일치)
* ``ready_after``: 몇 번 GET해야 RUNNING + ssh.direct가 되나 · ``boot_status``: 준비 중에 보일 상태(ERROR 등)
* ``create_mode="late"``: POST가 응답 없이 끊기고 pod는 ``late_after``번째 목록 조회부터 **늦게** 나타난다 (클라이언트가 포기한 뒤 RunPod가
  만든 경우 — 리뷰 1의 I2) · ``"never"``: 끊기고 끝내 나타나지 않는다
* ``external_delete_marker``: 이 파일이 생기면 그 pod는 지워진 것이다 — pod 안의 감시가 **자기 길로** 지운 것을 흉내 낸다(가짜 runpodctl이
  이 파일을 만든다)
* ``running``: 지움 확인 전 pod의 상태를 RUNNING으로 고정하고 싶을 때 쓰는 pods 사전을 검사가 직접 고친다
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any

CATALOG = [
    {"id": "NVIDIA H100 PCIe", "name": "H100 PCIe", "memory": 80, "secure": True, "community": True,
     "price": {"secure": 2.39, "community": 1.99}, "maxCount": {"secure": 8, "community": 2}},
    {"id": "NVIDIA H100 80GB HBM3", "name": "H100 SXM", "memory": 80, "secure": True, "community": True,
     "price": {"secure": 2.69, "community": 2.29}, "maxCount": {"secure": 8, "community": 8}},
    {"id": "NVIDIA H200", "name": "H200 SXM", "memory": 141, "secure": True, "community": False,
     "price": {"secure": 3.59, "community": 0.0}, "maxCount": {"secure": 8, "community": 0}},
]


class FakeTimeout(TimeoutError):
    """전송이 응답 없이 끊겼다 (OSError 계열 — 클라이언트는 POST에서 '애매함'으로 본다)."""


class FakeRunPodApi:
    def __init__(self, *, create_mode: str = "ok", delete_failures: int = 0, delete_never: bool = False, cost_override: float | None = None,
                 ready_after: int = 1, boot_status: str | None = None, host: str = "203.0.113.7", port: int = 40022) -> None:  # fmt: skip
        self.create_mode = create_mode
        self.delete_failures = int(delete_failures)
        self.delete_never = bool(delete_never)
        self.cost_override = cost_override
        self.ready_after = int(ready_after)
        self.boot_status = boot_status
        self.host, self.port = host, port
        self.pods: dict[str, dict[str, Any]] = {}
        self.gets: dict[str, int] = {}
        self.requests: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.volumes: list[dict[str, Any]] = []
        #: pod id → 청구액 (GET /billing/pods가 돌려준다)
        self.billed: dict[str, float] = {}
        self.late_after = 3
        self.list_calls = 0
        self.external_delete_marker: str | None = None
        self._late: dict[str, Any] | None = None
        self._next = 0

    # -- 기록에서 고르기 --

    def calls(self, method: str | None = None, prefix: str = "") -> list[dict[str, Any]]:
        return [r for r in self.requests if (method is None or r["method"] == method) and r["path"].startswith(prefix)]

    def live_pods(self) -> list[dict[str, Any]]:
        return [pod for pod in self.pods.values() if pod["status"] != "TERMINATED"]

    # -- 전송 --

    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> tuple[int, dict[str, str], bytes]:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.split("/v2/", 1)[-1]
        query = dict(urllib.parse.parse_qsl(parsed.query))
        payload = json.loads(body.decode("utf-8")) if body else None
        self.requests.append({"method": method, "path": path, "query": query, "body": payload, "headers": dict(headers), "url": url})
        if method == "GET" and path == "catalog/gpus":
            gpus = [dict(g) for g in CATALOG]
            if query.get("include") == "AVAILABILITY":
                for g in gpus:
                    g["availability"] = "HIGH"
                    g["cudaVersions"] = ["12.8", "13.0"]
            return self._json(200, {"gpus": gpus})
        self._external_delete()
        if method == "GET" and path == "pods":
            self.list_calls += 1
            if self._late is not None and self.list_calls >= self.late_after:
                self.pods[self._late["id"]] = self._late
                self._late = None
            return self._json(200, {"pods": [self._view(p) for p in self.pods.values() if p["status"] != "TERMINATED"],
                                    "pagination": {"nextCursor": None, "hasNextPage": False}})  # fmt: skip
        if method == "POST" and path == "pods":
            return self._create(payload)
        if path.startswith("pods/"):
            pod_id = urllib.parse.unquote(path.split("/", 1)[1])
            pod = self.pods.get(pod_id)
            if method == "GET":
                if pod is None or pod["status"] == "TERMINATED":
                    return self._json(404, {"title": "Not Found", "status": 404, "detail": "pod not found"})
                self.gets[pod_id] = self.gets.get(pod_id, 0) + 1
                if self.boot_status and pod["status"] != "RUNNING":
                    pod["status"] = self.boot_status
                elif self.gets[pod_id] >= self.ready_after and pod["status"] in ("PROVISIONING", "STARTING"):
                    pod["status"] = "RUNNING"
                return self._json(200, self._view(pod))
            if method == "DELETE":
                if pod is None or pod["status"] == "TERMINATED":
                    return self._json(404, {"title": "Not Found", "status": 404, "detail": "pod not found"})
                if self.delete_never or self.delete_failures > 0:
                    self.delete_failures -= 1
                    return self._json(500, {"title": "Internal", "status": 500, "detail": "terminate failed upstream"})
                pod["status"] = "TERMINATED"
                self.deleted.append(pod_id)
                return 204, {}, b""
        if method == "GET" and path == "network-volumes":
            return self._json(200, {"networkVolumes": self.volumes})
        if method == "GET" and path == "billing/pods":
            amount = float(self.billed.get(query.get("podId"), 0.0))
            return self._json(200, {"records": [{"podId": query.get("podId"), "totalAmount": amount}] if amount else [],
                                    "metadata": {"totals": {"totalAmount": amount, "gpuAmount": amount, "cpuAmount": 0.0, "diskAmount": 0.0}}})  # fmt: skip
        return self._json(404, {"title": "Not Found", "status": 404, "detail": f"no route {method} {path}"})

    # -- 안쪽 --

    @staticmethod
    def _json(status: int, data: Any) -> tuple[int, dict[str, str], bytes]:
        return status, {"Content-Type": "application/json"}, json.dumps(data).encode("utf-8")

    def _external_delete(self) -> None:
        """pod 안의 감시가 자기 길로 지웠다(가짜 runpodctl이 표지 파일을 만든다) — 그 pod들을 TERMINATED로."""
        import os

        if self.external_delete_marker and os.path.exists(self.external_delete_marker):
            for pod in self.pods.values():
                if pod["status"] != "TERMINATED":
                    pod["status"] = "TERMINATED"
                    pod["deleted_by"] = "in-pod"

    def _create(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        if self.create_mode == "timeout_before_create":
            raise FakeTimeout("read timed out")
        if self.create_mode in ("late", "never"):
            self.created.append(payload)
            if self.create_mode == "late":
                self._next += 1
                self._late = {"id": f"fakepod{self._next:03d}", "name": payload["name"], "status": "RUNNING", "cost": 2.69,
                              "env": dict(payload.get("env") or {}), "body": payload, "createdAt": "2026-10-11T12:00:00Z"}  # fmt: skip
            raise FakeTimeout("read timed out (RunPod은 아직 만드는 중)")
        if self.create_mode == "http_409":
            return self._json(409, {"title": "Conflict", "status": 409, "detail": "There are no longer any instances available with the requested specifications"})
        if self.create_mode == "http_500":
            return self._json(500, {"title": "Internal", "status": 500, "detail": "upstream error"})
        self._next += 1
        pod_id = f"fakepod{self._next:03d}"
        gpu = next(g for g in CATALOG if g["id"] == payload["gpu"]["id"])
        cost = gpu["price"][payload.get("cloud", "SECURE").lower()] * int(payload["gpu"].get("count", 1))
        pod = {"id": pod_id, "name": payload["name"], "status": "PROVISIONING", "cost": self.cost_override if self.cost_override is not None else cost,
               "env": dict(payload.get("env") or {}), "body": payload, "createdAt": "2026-10-11T12:00:00Z"}  # fmt: skip
        self.pods[pod_id] = pod
        self.created.append(payload)
        if self.create_mode == "timeout_after_create":
            raise FakeTimeout("connection reset after the request was sent")
        return self._json(201, self._view(pod))

    def _view(self, pod: dict[str, Any]) -> dict[str, Any]:
        running = pod["status"] == "RUNNING"
        return {
            "id": pod["id"], "name": pod["name"], "status": pod["status"], "cost": pod["cost"], "env": pod["env"],
            "image": pod["body"]["image"], "cloud": pod["body"].get("cloud"), "gpu": {"id": pod["body"]["gpu"]["id"], "count": pod["body"]["gpu"].get("count", 1)},
            "cudaVersion": "13.0", "dataCenterId": "US-TX-3", "createdAt": pod["createdAt"],
            "ssh": {"proxy": None, "direct": {"host": self.host, "port": self.port, "username": "root", "command": "ssh root@x"} if running else None},
        }  # fmt: skip
