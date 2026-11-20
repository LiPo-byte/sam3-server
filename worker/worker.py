"""家里的 GPU Worker。只出站连接云端，把请求原样转给本机服务，响应原样回传。

用户调 POST /api/gpu/<服务名>/<接口路径>，Worker 转到 <服务地址>/<接口路径>。
新服务：这里加一行 service(...)，云端 gpu_worker.SERVICES 登记同名服务和放行的接口路径。
同一服务加新接口只改云端白名单。

环境变量：
    GPU_WORKER_WS_URL    例如 wss://域名/api/gpu-worker/connect
    GPU_WORKER_TOKEN     与云端 CVAT_GPU_WORKER_TOKEN 相同
    WATERMARK_URL        wmdetect 根地址，例如 http://127.0.0.1:32102，不配则不接水印
    SAM3_URL             sam3 根地址，例如 http://127.0.0.1:32100，不配则不接 SAM3
"""

import asyncio
import logging
import os

from core import SERVICES, logger, service, session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

service("watermark", os.environ.get("WATERMARK_URL", ""))
service("sam3", os.environ.get("SAM3_URL", ""))


async def main() -> None:
    ws_url = os.environ.get("GPU_WORKER_WS_URL", "").strip()
    token = os.environ.get("GPU_WORKER_TOKEN", "").strip()
    if not ws_url:
        raise SystemExit("需要 GPU_WORKER_WS_URL")
    if not token:
        raise SystemExit("需要 GPU_WORKER_TOKEN")
    if not SERVICES:
        raise SystemExit("没有可用服务，至少配置 WATERMARK_URL 或 SAM3_URL")
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
