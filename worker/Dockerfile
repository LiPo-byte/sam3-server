FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt -i ${PIP_INDEX_URL}

COPY worker.py core.py ./

CMD ["python", "worker.py"]
