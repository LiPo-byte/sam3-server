"""Worker 框架：连云端、上报服务、原样转发，每个服务单独限并发。

云端下发 {service, path, content_type, blob_path, timeout}。Worker 下载请求体，
原样 POST 到 本机服务地址 + path，把状态码、Content-Type、响应体原样回传，不解析内容。
"""

import asyncio
import base64
import inspect
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger("gpu-worker")

FETCH_TIMEOUT = 30
DEFAULT_TIMEOUT = 50


@dataclass(frozen=True)
class Service:
    name: str
    base_url: str
    concurrency: int


SERVICES: dict[str, Service] = {}


def service(name: str, base_url: str, concurrency: int = 1) -> None:
    """base_url 为空则不登记，也不会在 hello 里上报，云端对该服务直接回 503。"""
    base_url = (base_url or "").strip().rstrip("/")
    if base_url:
        SERVICES[name] = Service(name, base_url, concurrency)


def http_origin(ws_url: str) -> str:
    if ws_url.startswith("wss://"):
        scheme = "https://"
        rest = ws_url[len("wss://"):]
    elif ws_url.startswith("ws://"):
        scheme = "http://"
        rest = ws_url[len("ws://"):]
    else:
        raise ValueError("GPU_WORKER_WS_URL 需要 ws:// 或 wss://")
    host = rest.split("/", 1)[0]
    if not host:
        raise ValueError("GPU_WORKER_WS_URL 缺少主机名")
    return scheme + host


def download_blob(origin: str, blob_path: str, token: str) -> bytes:
    request = urllib.request.Request(
        origin.rstrip("/") + blob_path,
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"blob http {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("blob unreachable") from exc


def forward(spec: Service, path: str, body: bytes, content_type: str, timeout: float) -> tuple[int, str, bytes]:
    """原样 POST 给本机服务。4xx/5xx 也是服务的正常响应，原样返回；连不上才算失败。"""
    request = urllib.request.Request(
        spec.base_url + path,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{spec.name} unreachable") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{spec.name} timed out") from exc


def run_task(origin: str, token: str, message: dict) -> dict:
    job_id = message["id"]
    name = message.get("service")
    path = message.get("path") or ""
    started = time.monotonic()
    spec = SERVICES.get(name)
    if spec is None or not path.startswith("/"):
        logger.warning("job %s unknown service %s%s", job_id, name, path)
        return {"type": "result", "id": job_id, "ok": False, "error": f"unknown service {name}{path}"}
    try:
        body = download_blob(origin, message["blob_path"], token)
        status, content_type, data = forward(
            spec, path, body,
            message.get("content_type") or "application/json",
            float(message.get("timeout") or DEFAULT_TIMEOUT),
        )
    except Exception as exc:
        logger.warning("job %s %s%s failed after %.0fms: %s",
                       job_id, name, path, (time.monotonic() - started) * 1000, exc)
        return {"type": "result", "id": job_id, "ok": False, "error": str(exc)}
    logger.info("job %s %s%s status=%s %dB %.0fms",
                job_id, name, path, status, len(data), (time.monotonic() - started) * 1000)
    result = {"type": "result", "id": job_id, "ok": True, "status": status, "content_type": content_type}
    try:
        result["body"] = data.decode("utf-8")
    except UnicodeDecodeError:
        result["body_b64"] = base64.b64encode(data).decode()
    return result


async def session(ws_url: str, token: str) -> None:
    import websockets

    origin = http_origin(ws_url)
    headers = {"Authorization": f"Bearer {token}"}
    params = inspect.signature(websockets.connect).parameters
    header_key = "additional_headers" if "additional_headers" in params else "extra_headers"
    connect_kwargs = {
        header_key: headers,
        "ping_interval": 20,
        "ping_timeout": 20,
        "max_size": 2 * 1024 * 1024,
        "open_timeout": 10,
    }
    limits = {name: asyncio.Semaphore(spec.concurrency) for name, spec in SERVICES.items()}
    running: set[asyncio.Task] = set()

    async with websockets.connect(ws_url, **connect_kwargs) as ws:
        logger.info("connected %s services=%s", ws_url, sorted(SERVICES))
        await ws.send(json.dumps({
            "type": "hello",
            "services": {name: spec.concurrency for name, spec in SERVICES.items()},
        }))

        async def handle(message: dict) -> None:
            limit = limits.get(message.get("service")) or asyncio.Semaphore(1)
            try:
                async with limit:
                    result = await asyncio.to_thread(run_task, origin, token, message)
                await ws.send(json.dumps(result))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("job %s send failed: %s", message.get("id"), exc)

        async def recv_loop() -> None:
            async for raw in ws:
                message = json.loads(raw)
                if message.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                    continue
                if message.get("type") == "task":
                    job = asyncio.create_task(handle(message))
                    running.add(job)
                    job.add_done_callback(running.discard)

        async def ping_loop() -> None:
            while True:
                await asyncio.sleep(15)
                await ws.send(json.dumps({"type": "ping"}))

        tasks = [
            asyncio.create_task(recv_loop()),
            asyncio.create_task(ping_loop()),
        ]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            pending_all = [t for t in tasks if not t.done()] + list(running)
            for t in pending_all:
                t.cancel()
            await asyncio.gather(*pending_all, return_exceptions=True)
        for t in done:
            exc = t.exception()
            if exc:
                raise exc
