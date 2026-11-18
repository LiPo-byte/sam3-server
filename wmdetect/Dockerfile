ARG BASE_IMAGE=sam3-base:torch2.11-cu128
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

# torch comes from the base image; pin it so no requirement can upgrade or downgrade it
COPY requirements.txt /tmp/requirements.txt
RUN pip freeze | grep -iE '^(torch|torchvision|triton|nvidia-)' > /tmp/torch-constraints.txt \
 && pip install -r /tmp/requirements.txt -c /tmp/torch-constraints.txt -i ${PIP_INDEX_URL}

WORKDIR /srv
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
