FROM python:3.12-slim-bookworm

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu126/

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# opencv-contrib (pulled by paddlex) needs the GL/X libs; paddle needs libgomp
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 libxrender1 curl \
 && rm -rf /var/lib/apt/lists/*

# paddle GPU wheel is not on PyPI; install it before paddleocr so the CPU package is not pulled
COPY requirements.txt /tmp/requirements.txt
RUN pip install "paddlepaddle-gpu==3.2.0" -i "${PADDLE_INDEX_URL}" \
 && pip install -r /tmp/requirements.txt -i "${PIP_INDEX_URL}"

WORKDIR /srv
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
