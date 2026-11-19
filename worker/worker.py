"""家里的 GPU Worker。只出站连接云端，按任务名转给本机服务。

目前只有水印。新任务在 TASKS 里注册，函数返回要放进 result.body 的字典。
消息不带 task 时仍按水印处理。

环境变量：
    GPU_WORKER_WS_URL    例如 wss://域名/api/gpu-worker/connect
    GPU_WORKER_TOKEN     与云端 CVAT_GPU_WORKER_TOKEN 相同
    WATERMARK_URL        默认 http://127.0.0.1:32102/detect
"""

import asyncio
import base64
import inspect
import json
import logging
import os
import time
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gpu-worker")

WATERMARK_URL = os.environ.get("WATERMARK_URL", "http://127.0.0.1:32102/detect")
LOCAL_TIMEOUT = 30


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


def detect_image(image: bytes, url: str) -> bool:
    payload = json.dumps({"image": base64.b64encode(image).decode()}).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=LOCAL_TIMEOUT) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"watermark http {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("watermark unreachable") from exc
    if "watermarked" not in data:
        raise RuntimeError("watermark response missing watermarked")
    return bool(data["watermarked"])


def download_blob(origin: str, blob_path: str, token: str) -> bytes:
    request = urllib.request.Request(
        origin.rstrip("/") + blob_path,
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=LOCAL_TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"blob http {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("blob unreachable") from exc


def run_watermark(origin: str, token: str, message: dict) -> dict:
    image = download_blob(origin, message["blob_path"], token)
    return {"has_watermark": detect_image(image, WATERMARK_URL)}


# 新任务在这里注册。云端消息带 {"type":"task","task":"<名字>",...}。
TASKS = {
    "watermark": run_watermark,
}


def run_task(origin: str, token: str, message: dict) -> dict:
    job_id = message["id"]
    name = message.get("task") or "watermark"
    started = time.monotonic()
    handler = TASKS.get(name)
    if handler is None:
        logger.warning("job %s unknown task %s", job_id, name)
        return {"type": "result", "id": job_id, "ok": False, "error": f"unknown task {name}"}
    try:
        body = handler(origin, token, message)
    except Exception as exc:
        logger.warning("job %s failed after %.0fms: %s", job_id, (time.monotonic() - started) * 1000, exc)
        return {"type": "result", "id": job_id, "ok": False, "error": str(exc)}
    if name == "watermark":
        logger.info(
            "job %s has_watermark=%s %.0fms",
            job_id,
            body.get("has_watermark"),
            (time.monotonic() - started) * 1000,
        )
    else:
        logger.info("job %s task=%s %.0fms", job_id, name, (time.monotonic() - started) * 1000)
    return {"type": "result", "id": job_id, "ok": True, "body": body}


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
    queue: asyncio.Queue = asyncio.Queue()

    async with websockets.connect(ws_url, **connect_kwargs) as ws:
        logger.info("connected %s", ws_url)

        async def recv_loop() -> None:
            async for raw in ws:
                message = json.loads(raw)
                if message.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                    continue
                if message.get("type") == "task":
                    await queue.put(message)

        async def work_loop() -> None:
            while True:
                message = await queue.get()
                result = await asyncio.to_thread(run_task, origin, token, message)
                await ws.send(json.dumps(result))

        async def ping_loop() -> None:
            while True:
                await asyncio.sleep(15)
                await ws.send(json.dumps({"type": "ping"}))

        tasks = [
            asyncio.create_task(recv_loop()),
            asyncio.create_task(work_loop()),
            asyncio.create_task(ping_loop()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            exc = task.exception()
            if exc:
                raise exc


async def main() -> None:
    ws_url = os.environ.get("GPU_WORKER_WS_URL", "").strip()
    token = os.environ.get("GPU_WORKER_TOKEN", "X2t8Lk5Bq9Rp").strip()
    if not ws_url:
        raise SystemExit("需要 GPU_WORKER_WS_URL")
    delay = 1
    while True:
        try:
            await session(ws_url, token)
            delay = 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("connection lost: %s", exc)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30)


if __name__ == "__main__":
    asyncio.run(main())
