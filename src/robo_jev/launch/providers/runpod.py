"""RunPod 공급자 — REST API v2(`https://api.runpod.io/v2`)로 pod를 만들고·보고·지운다 (Task cloud1).

의존성을 더하지 않는다: 표준 라이브러리 ``urllib``만 쓴다(공식 SDK는 GraphQL 위에 있고 v1 REST는 2026-11-15에 퇴역한다).
전송(`transport`)은 주입할 수 있어 검사는 가짜 API(`tests/runpod_fake.py`)로 같은 코드를 돈다.

**돈 안전의 자리 (이 파일이 지키는 것).**

* **키는 파일 하나에서만 읽는다** — :data:`DEFAULT_KEY_FILE`(`~/.config/robo-jev/runpod.env`, git 밖). 환경 변수·명령 인자로는
  받지 않는다(프로세스 목록과 셸 기록에 남는다). 그룹·다른 사용자가 읽을 수 있는 파일은 거절한다. 값은 :class:`Secret`에 담겨
  `repr`·`str`·서식에서 언제나 ``***``이고, 오류 문장은 :func:`redact`를 거친다 — 키가 로그·명세·예외에 새지 않는다.
* **읽기 전용 클라이언트**(`read_only=True`)는 GET 밖의 요청을 **보내기 전에** 거절한다(:class:`ReadOnlyViolation`) — dry-run·가격
  조회·자원 목록이 실수로 pod를 만들 수 없다.
* **만들기(POST)는 재시도하지 않는다** — 응답을 못 받은 POST를 다시 보내면 pod가 둘이 되고 돈도 두 배다. 429(서버가 처리하지 않았다는
  뜻)만 기다렸다 다시 보낸다. 응답이 애매하게 끊기면 호출자가 **이름으로 찾아 지운다**(:meth:`RunPodProvider.sweep_name`).
* **단가 대조** — 만들기 전에 카탈로그의 GPU 단가 × 수를 명세의 `budget.hourly_usd`와 견주고, 만든 뒤에는 그 pod의 `cost`(실제
  청구 단가)를 다시 견준다. 다르면 거절이고, 이미 만들었으면 **곧바로 지운다**.
* **지우기는 확인될 때까지** — DELETE가 204를 돌려줘도 GET이 404(또는 `TERMINATED`)를 줄 때까지 확인한다. 확인하지 못하면 거짓을
  돌려주고 실행기가 `unknown`을 남긴다(docs/06 Task 6: 종료 API 실패는 미종료 상태로 명시).
"""

from __future__ import annotations

import json
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "API_BASE",
    "DEFAULT_KEY_FILE",
    "KEY_NAME",
    "MissingApiKey",
    "PriceMismatch",
    "ReadOnlyViolation",
    "RunPodClient",
    "RunPodError",
    "RunPodProvider",
    "Secret",
    "UnsafeKeyFile",
    "load_api_key",
    "pod_request",
    "redact",
]

API_BASE = "https://api.runpod.io/v2"
DEFAULT_KEY_FILE = Path("~/.config/robo-jev/runpod.env")
KEY_NAME = "RUNPOD_API_KEY"
#: 단가를 견줄 때의 허용 오차(USD/h) — 반 센트. 콘솔·카탈로그는 센트 단위로 적는다.
PRICE_TOLERANCE = 0.005
#: 살아 있는 상태 — 이 상태의 pod는 돈이 나간다(PROVISIONING도 자원을 붙잡고 있다고 본다).
LIVE_STATUSES = ("PROVISIONING", "STARTING", "RUNNING")


class MissingApiKey(RuntimeError):
    """키 파일이 없거나 비었다 — pod를 만들지 않는다."""


class UnsafeKeyFile(RuntimeError):
    """키 파일을 그룹·다른 사용자가 읽을 수 있다 — `chmod 600`이 먼저다."""


class ReadOnlyViolation(RuntimeError):
    """읽기 전용 클라이언트에 GET이 아닌 요청이 왔다 — 보내지 않았다."""


class PriceMismatch(RuntimeError):
    """API의 시간 단가가 명세의 `budget.hourly_usd`와 다르다."""


class RunPodError(RuntimeError):
    """API 오류. `status`는 HTTP 상태(전송 실패는 0), `ambiguous`는 '보냈는지 모른다'(POST의 응답을 못 받음)."""

    def __init__(self, message: str, *, status: int = 0, title: str | None = None, detail: str | None = None, ambiguous: bool = False) -> None:
        super().__init__(message)
        self.status = int(status)
        self.title = title
        self.detail = detail
        self.ambiguous = bool(ambiguous)


class Secret:
    """API 키를 감싼다. `repr`·`str`·서식은 언제나 ``***`` — 실수로 찍혀도 새지 않는다. 값은 :meth:`reveal`로만 꺼낸다."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = str(value)

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    def __str__(self) -> str:
        return "***"

    def __format__(self, spec: str) -> str:
        return "***"

    def __reduce__(self) -> Any:  # pickle·copy로도 값이 나가지 않게
        raise TypeError("Secret은 직렬화하지 않는다")


def redact(text: Any, secret: Secret | None) -> str:
    """문장 안의 키를 ``***``로 바꾼다 (오류 메시지·로그로 나가기 전에)."""
    out = str(text)
    if secret is not None and secret.reveal():
        out = out.replace(secret.reveal(), "***")
    return out


def _parse_key_text(text: str, path: Path) -> str:
    """키 파일의 두 형식: ``RUNPOD_API_KEY=<값>``(앞의 ``export``·따옴표 허용) 또는 맨 키 한 줄. 주석(#)·빈 줄은 건너뛴다."""
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise MissingApiKey(f"{path}: 비었다 — `{KEY_NAME}=<키>` 한 줄(또는 키만 한 줄)을 적는다")
    assigned = []
    for line in lines:
        body = line[len("export "):].strip() if line.startswith("export ") else line
        name, sep, value = body.partition("=")
        if sep and name.strip() == KEY_NAME:
            assigned.append(value.strip())
    if assigned:
        if len(assigned) != 1:
            raise MissingApiKey(f"{path}: `{KEY_NAME}=` 줄이 {len(assigned)}개다 — 하나만 둔다")
        value = assigned[0]
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
    elif len(lines) == 1 and "=" not in lines[0]:
        value = lines[0]
    else:
        raise MissingApiKey(f"{path}: `{KEY_NAME}=<키>` 줄이 없다 (맨 키는 한 줄만 받는다)")
    if not value or any(ch.isspace() for ch in value) or not value.isprintable() or len(value) < 8:
        raise MissingApiKey(f"{path}: 키 값의 꼴이 이상하다 (비었거나 공백·제어 문자가 있거나 너무 짧다) — 값은 찍지 않는다")
    return value


def load_api_key(path: str | Path | None = None) -> Secret:
    """git 밖의 키 파일에서 API 키를 읽는다. **환경 변수·인자는 보지 않는다.**

    없으면 :class:`MissingApiKey`, 그룹·다른 사용자가 읽을 수 있으면 :class:`UnsafeKeyFile`(값을 읽기 전에 거절한다).
    """
    target = Path(path if path is not None else DEFAULT_KEY_FILE).expanduser()
    if not target.is_file():
        raise MissingApiKey(
            f"{target}: RunPod API 키 파일이 없다 — 콘솔(Settings → API Keys)에서 만든 키를 `{KEY_NAME}=<키>` 한 줄로 적고 "
            "`chmod 600`한다. pod를 만들지 않는다"
        )
    mode = target.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise UnsafeKeyFile(
            f"{target}: 권한이 {stat.filemode(mode)}다 — 그룹·다른 사용자가 읽을 수 있는 키 파일은 쓰지 않는다. `chmod 600 {target}` 뒤 다시"
        )
    return Secret(_parse_key_text(target.read_text(encoding="utf-8"), target))


# --------------------------------------------------------------------------
# 전송
# --------------------------------------------------------------------------

#: (method, url, headers, body, timeout) → (status, headers, body). 전송 실패는 OSError 계열을 그대로 낸다.
Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, dict[str, str], bytes]]


def urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> tuple[int, dict[str, str], bytes]:
    """표준 라이브러리 전송. HTTP 오류 상태도 (status, headers, body)로 돌려준다(예외가 아니다)."""
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 — URL은 API_BASE에서만 만든다
            return int(response.status), dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), dict(error.headers.items()) if error.headers else {}, error.read() or b""


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]
    data: Any


class RunPodClient:
    """REST API v2의 얇은 클라이언트. 쓰기는 `read_only=False`일 때만 나간다."""

    def __init__(
        self,
        key: Secret,
        *,
        transport: Transport | None = None,
        base: str = API_BASE,
        read_only: bool = False,
        timeout: float = 30.0,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(key, Secret):
            raise TypeError("key는 Secret이어야 한다 (load_api_key가 만든다)")
        self._key = key
        self._transport = transport or urllib_transport
        self.base = base.rstrip("/")
        self.read_only = bool(read_only)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._sleep = sleep
        #: 보낸 요청의 장부 — (method, path). 헤더·본문은 적지 않는다(키가 헤더에 있다).
        self.calls: list[tuple[str, str]] = []

    def __repr__(self) -> str:
        return f"RunPodClient(base={self.base!r}, read_only={self.read_only})"

    def _url(self, path: str, query: dict[str, Any] | None) -> str:
        url = f"{self.base}/{path.lstrip('/')}"
        if query:
            pairs = {key: (",".join(value) if isinstance(value, (list, tuple)) else value) for key, value in query.items() if value is not None}
            url += "?" + urllib.parse.urlencode(pairs)
        return url

    def request(self, method: str, path: str, *, query: dict[str, Any] | None = None, body: Any = None) -> Response:
        method = method.upper()
        if self.read_only and method != "GET":
            raise ReadOnlyViolation(f"읽기 전용 클라이언트가 {method} {path}를 받았다 — 보내지 않았다")
        url = self._url(path, query)
        headers = {"Authorization": f"Bearer {self._key.reveal()}", "Accept": "application/json"}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        idempotent = method in ("GET", "DELETE", "PUT")
        attempt = 0
        while True:
            attempt += 1
            self.calls.append((method, path))
            try:
                status, response_headers, raw = self._transport(method, url, headers, payload, self.timeout)
            except OSError as error:  # 연결 실패·시간 초과 — POST는 '보냈는지 모른다'
                if idempotent and attempt <= self.max_retries:
                    self._sleep(min(2.0 ** attempt, 30.0))
                    continue
                raise RunPodError(
                    f"{method} {path}: 전송 실패 ({type(error).__name__}: {redact(error, self._key)})",
                    status=0, ambiguous=not idempotent,
                ) from None  # fmt: skip
            if status == 429 and attempt <= self.max_retries:
                # 429는 서버가 요청을 처리하지 않았다는 뜻 — POST도 기다렸다 다시 보낸다.
                self._sleep(_retry_after(response_headers, attempt))
                continue
            if status >= 500 and idempotent and attempt <= self.max_retries:
                self._sleep(min(2.0 ** attempt, 30.0))
                continue
            data = _decode(raw)
            if status >= 400:
                title = detail = None
                if isinstance(data, dict):
                    title, detail = data.get("title"), data.get("detail")
                    if data.get("errors"):
                        detail = f"{detail} {data.get('errors')}"
                raise RunPodError(
                    redact(f"{method} {path}: HTTP {status} {title or ''} — {detail or str(data)[:300]}", self._key),
                    status=status, title=title, detail=redact(detail, self._key) if detail else None,
                    ambiguous=(status >= 500 and not idempotent),
                )  # fmt: skip
            return Response(status=status, headers=response_headers, data=data)

    # -- 읽기 --

    def gpu_types(self, *, availability: bool = False, cloud: str | None = None, count: int | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if availability:
            query = {"include": "AVAILABILITY", "product": "POD", "cloud": cloud, "count": count}
        return list((self.request("GET", "catalog/gpus", query=query).data or {}).get("gpus") or [])

    def list_pods(self) -> list[dict[str, Any]]:
        pods: list[dict[str, Any]] = []
        cursor = None
        for _ in range(100):  # 1000개씩 — 끝없는 고리를 막는다
            data = self.request("GET", "pods", query={"cursor": cursor}).data or {}
            pods.extend(data.get("pods") or [])
            page = data.get("pagination") or {}
            if not page.get("hasNextPage") or not page.get("nextCursor"):
                break
            cursor = page["nextCursor"]
        return pods

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        """pod 하나. 없으면(404) None."""
        try:
            return self.request("GET", f"pods/{urllib.parse.quote(pod_id, safe='')}").data
        except RunPodError as error:
            if error.status == 404:
                return None
            raise

    def list_network_volumes(self) -> list[dict[str, Any]]:
        data = self.request("GET", "network-volumes").data
        if isinstance(data, dict):
            return list(data.get("networkVolumes") or data.get("volumes") or data.get("items") or [])
        return list(data or [])

    def pod_billing(self, *, pod_id: str | None = None, start: str | None = None, end: str | None = None, bucket: str = "hour") -> dict[str, Any]:
        query = {"podId": pod_id, "startTime": start, "endTime": end, "bucketSize": bucket}
        return self.request("GET", "billing/pods", query=query).data or {}

    # -- 쓰기 (read_only면 보내기 전에 거절) --

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "pods", body=body).data

    def delete_pod(self, pod_id: str) -> int:
        """DELETE /pods/{id}. 돌려주는 것은 상태 코드(204). 이미 없으면(404) 404를 돌려준다(오류가 아니다)."""
        try:
            return self.request("DELETE", f"pods/{urllib.parse.quote(pod_id, safe='')}").status
        except RunPodError as error:
            if error.status == 404:
                return 404
            raise


def _decode(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw.decode("utf-8", "replace")[:500]


def _retry_after(headers: dict[str, str], attempt: int) -> float:
    for name, value in headers.items():
        if name.lower() == "retry-after":
            try:
                return min(max(float(value), 1.0), 60.0)
            except ValueError:
                break
    return min(2.0 ** attempt, 30.0)


# --------------------------------------------------------------------------
# pod 요청 본문
# --------------------------------------------------------------------------


def pod_request(spec: dict[str, Any], *, name: str, env: dict[str, str], start_script: str | None) -> dict[str, Any]:
    """명세의 `provider` 블록 → `POST /v2/pods` 본문 (순수 함수 — 검사가 그대로 본다).

    GPU 종류·수·클라우드(SECURE/COMMUNITY)·이미지·컨테이너 디스크·볼륨·지역·CUDA 하한이 명세에서 온다. `start_script`가 있으면
    컨테이너를 ``/bin/sh -c <script>``로 띄운다 — 자기 종료 감시를 먼저 띄우고 이미지의 원래 시작(``/start.sh``: sshd)을 `exec`한다.
    """
    gpu: dict[str, Any] = {"id": str(spec["gpu_type"]), "count": int(spec.get("gpu_count") or 1)}
    if spec.get("min_cuda_version"):
        gpu["minCudaVersion"] = str(spec["min_cuda_version"])
    if spec.get("allowed_cuda_versions"):
        gpu["allowedCudaVersions"] = [str(v) for v in spec["allowed_cuda_versions"]]
    if spec.get("min_ram_per_gpu"):
        gpu["minRamPerGpu"] = int(spec["min_ram_per_gpu"])
    if spec.get("min_vcpu_per_gpu"):
        gpu["minVcpuCountPerGpu"] = int(spec["min_vcpu_per_gpu"])
    body: dict[str, Any] = {
        "name": name,
        "image": str(spec["image"]),
        "cloud": str(spec.get("cloud") or "SECURE").upper(),
        "gpu": gpu,
        "disk": int(spec.get("disk_gb") or 50),
        "ports": ["22/tcp"],
        "env": {str(k): str(v) for k, v in env.items()},
        "startSsh": True,
    }
    if spec.get("data_centers"):
        body["dataCenterIds"] = [str(dc) for dc in spec["data_centers"]]
    if spec.get("network_volume_id"):
        body["mounts"] = {"network": [{"volumeId": str(spec["network_volume_id"]), "path": str(spec.get("volume_path") or "/workspace")}]}
    elif spec.get("volume_gb"):
        body["mounts"] = {"persistent": {"size": int(spec["volume_gb"]), "path": str(spec.get("volume_path") or "/workspace")}}
    if start_script:
        body["entrypoint"] = ["/bin/sh", "-c"]
        body["cmd"] = [start_script]
    return body


# --------------------------------------------------------------------------
# 공급자
# --------------------------------------------------------------------------


@dataclass
class PodView:
    """pod 응답에서 실행기가 쓰는 것만."""

    pod_id: str
    status: str | None
    cost: float | None
    host: str | None
    port: int | None
    username: str | None
    cuda_version: str | None
    data_center: str | None
    raw: dict[str, Any]

    @classmethod
    def of(cls, pod: dict[str, Any]) -> PodView:
        direct = ((pod.get("ssh") or {}).get("direct")) or {}
        cost = pod.get("cost")
        return cls(
            pod_id=str(pod.get("id")),
            status=pod.get("status"),
            cost=None if cost is None else float(cost),
            host=direct.get("host"),
            port=None if direct.get("port") is None else int(direct["port"]),
            username=direct.get("username"),
            cuda_version=pod.get("cudaVersion"),
            data_center=pod.get("dataCenterId"),
            raw=pod,
        )


class RunPodProvider:
    """pod의 수명 — `Provider` 인터페이스(create/destroy/describe)에 돈 안전 검사를 더한 것."""

    name = "runpod"

    def __init__(self, client: RunPodClient, *, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self.client = client
        self._sleep = sleep
        self._clock = clock

    # -- 단가 --

    def node_price(self, gpu_type: str, cloud: str, count: int) -> tuple[float, dict[str, Any]]:
        """카탈로그의 GPU 한 장 단가 × 수 = 노드 시간 단가. 카탈로그에 없거나 그 클라우드에 없으면 :class:`RunPodError`."""
        for gpu in self.client.gpu_types():
            if gpu.get("id") == gpu_type:
                cloud_key = str(cloud).lower()
                if not gpu.get(cloud_key, False):
                    raise RunPodError(f"{gpu_type}: {cloud} 클라우드에서 내놓지 않는 GPU다 (카탈로그 secure={gpu.get('secure')}, community={gpu.get('community')})")
                price = (gpu.get("price") or {}).get(cloud_key)
                if price is None:
                    raise RunPodError(f"{gpu_type}: 카탈로그에 {cloud} 단가가 없다")
                return round(float(price) * int(count), 6), gpu
        raise RunPodError(f"{gpu_type}: 카탈로그(GET /v2/catalog/gpus)에 없는 GPU id다")

    def check_price(self, spec: dict[str, Any], expected_hourly_usd: float) -> dict[str, Any]:
        """만들기 **전의** 단가 대조. 다르면 :class:`PriceMismatch` — pod를 만들지 않는다."""
        price, gpu = self.node_price(str(spec["gpu_type"]), str(spec.get("cloud") or "SECURE"), int(spec.get("gpu_count") or 1))
        if abs(price - float(expected_hourly_usd)) > PRICE_TOLERANCE:
            raise PriceMismatch(
                f"{spec['gpu_type']} × {spec.get('gpu_count') or 1} ({spec.get('cloud') or 'SECURE'}): 카탈로그 단가 ${price:.4f}/h ≠ 명세 "
                f"budget.hourly_usd ${float(expected_hourly_usd):.4f}/h — 명세를 고치고(--hourly-usd) 다시 prepare한다. pod를 만들지 않았다"
            )
        return {"catalog_hourly_usd": price, "gpu": {k: gpu.get(k) for k in ("id", "name", "memory", "secure", "community", "price")}}

    # -- 만들기·보기·지우기 --

    def create(self, body: dict[str, Any]) -> PodView:
        """POST /v2/pods. 재시도는 429뿐이다(:meth:`RunPodClient.request`). 응답이 애매하면 :class:`RunPodError`(ambiguous)."""
        pod = self.client.create_pod(body)
        if not isinstance(pod, dict) or not pod.get("id"):
            raise RunPodError(f"POST /v2/pods: 응답에 pod id가 없다 ({str(pod)[:200]})", ambiguous=True)
        return PodView.of(pod)

    def describe(self, pod_id: str) -> dict[str, Any] | None:
        return self.client.get_pod(pod_id)

    def check_created_price(self, view: PodView, expected_hourly_usd: float) -> None:
        """만든 **뒤의** 단가 대조 — pod의 `cost`가 실제 청구 단가다. 다르면 :class:`PriceMismatch`(호출자가 곧바로 지운다)."""
        if view.cost is None:
            return  # 아직 단가가 붙지 않았다 — 준비를 기다리는 동안 다시 본다
        if view.cost > 0 and abs(view.cost - float(expected_hourly_usd)) > PRICE_TOLERANCE:
            raise PriceMismatch(
                f"pod {view.pod_id}: 실제 청구 단가 ${view.cost:.4f}/h ≠ 명세 ${float(expected_hourly_usd):.4f}/h — 이 pod를 곧바로 지운다"
            )

    def wait_ready(self, pod_id: str, *, timeout: float, poll: float = 5.0, expected_hourly_usd: float | None = None,
                   on_poll: Callable[[PodView], None] | None = None) -> PodView:  # fmt: skip
        """RUNNING이고 `ssh.direct`가 생길 때까지. 기한을 넘기거나 EXITED·ERROR·TERMINATED가 되면 :class:`RunPodError`."""
        deadline = self._clock() + float(timeout)
        while True:
            pod = self.client.get_pod(pod_id)
            if pod is None:
                raise RunPodError(f"pod {pod_id}: 준비를 기다리는 사이에 사라졌다 (404)")
            view = PodView.of(pod)
            if on_poll is not None:
                on_poll(view)
            if expected_hourly_usd is not None:
                self.check_created_price(view, expected_hourly_usd)
            if view.status in ("EXITED", "ERROR", "TERMINATED"):
                raise RunPodError(f"pod {pod_id}: 준비 중에 {view.status}가 됐다")
            if view.status == "RUNNING" and view.host and view.port:
                return view
            if self._clock() >= deadline:
                raise RunPodError(f"pod {pod_id}: {timeout:.0f} s 안에 RUNNING + ssh.direct가 되지 않았다 (지금 {view.status})")
            self._sleep(poll)

    def destroy(self, pod_id: str, *, confirm_timeout: float = 180.0, poll: float = 5.0) -> bool:
        """지우고 **확인한다**. GET이 404이거나 TERMINATED일 때만 참이다. DELETE가 실패해도 기한까지 다시 보낸다."""
        deadline = self._clock() + float(confirm_timeout)
        while True:
            try:
                self.client.delete_pod(pod_id)
            except RunPodError:
                pass  # 아래 GET이 판정한다 — 실패한 DELETE도 기한 안에서는 다시 보낸다
            try:
                pod = self.client.get_pod(pod_id)
            except RunPodError:
                pod = {"status": "UNKNOWN"}
            if pod is None or pod.get("status") == "TERMINATED":
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(poll)

    def find_by_name(self, name: str) -> list[dict[str, Any]]:
        return [pod for pod in self.client.list_pods() if pod.get("name") == name]

    def sweep_name(self, name: str, *, confirm_timeout: float = 180.0) -> list[tuple[str, bool]]:
        """이 이름의 pod를 모두 지운다 — 응답을 못 받은 POST가 실제로는 pod를 만들었을 때의 뒷정리."""
        return [(str(pod["id"]), self.destroy(str(pod["id"]), confirm_timeout=confirm_timeout)) for pod in self.find_by_name(name)]


def key_file_permissions_note(path: str | Path | None = None) -> str | None:
    """키 파일이 든 디렉터리가 그룹·다른 사용자에게 쓰기 가능하면 경고 문장 (거절은 아니다 — 파일 자체는 600이다)."""
    target = Path(path if path is not None else DEFAULT_KEY_FILE).expanduser()
    try:
        mode = target.parent.stat().st_mode
    except OSError:
        return None
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        return f"경고: {target.parent}가 {stat.filemode(mode)}다 — 다른 사용자가 키 파일을 바꿔 놓을 수 있다 (`chmod 700 {target.parent}` 권장)"
    return None


def env_key_present() -> bool:
    """환경 변수에 키가 있는지(쓰지는 않는다) — 있으면 '파일만 읽는다'는 사실을 알린다."""
    return bool(os.environ.get(KEY_NAME))
