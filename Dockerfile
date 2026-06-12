# syntax=docker/dockerfile:1
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    libusb-1.0-0 libudev1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# open3d PyPI aarch64 wheels top out at 0.16.0; all APIs used here are compatible.
# Cache mount keeps downloaded wheels across build attempts so a flaky link can
# resume instead of re-fetching (the big aarch64 wheels otherwise hit ReadTimeoutError).
RUN --mount=type=cache,target=/root/.cache/pip \
    sed 's/open3d==0.17.0/open3d==0.16.0/' requirements.txt > /tmp/req.txt \
    && pip3 install --prefer-binary --timeout 300 --retries 10 -r /tmp/req.txt

COPY . .

RUN mkdir -p data/logs data/templates

CMD ["python3", "main.py"]
 