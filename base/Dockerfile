# Base image reusing torch from an existing Python 3.12 venv (see build.sh).
# Python minor version here must match the source venv.
FROM python:3.12-slim-bookworm

COPY . /usr/local/lib/python3.12/site-packages/

RUN python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchvision', torchvision.__version__)" \
 && (pip check || true)
