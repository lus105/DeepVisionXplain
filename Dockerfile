ARG PYTORCH_VERSION="2.7.1"
ARG CUDA_VERSION="12.6"
ARG CUDNN_VERSION="9"

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Vilnius

# ------------------------------ python checks ------------------------------ #

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ------------------------------ working directory -------------------------- #

WORKDIR /app
COPY . /app

# --------------------------------- packages -------------------------------- #

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        vim \
        htop \
        iotop \
        dos2unix \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------- timezone setup --------------------------- #

RUN ln -fs /usr/share/zoneinfo/Europe/Vilnius /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

ENV TZ=Europe/Vilnius

# ------------------------------- user & group ----------------------------- #

RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app \
    && dos2unix /app/scripts/startup_service.sh \
    && chmod +x /app/scripts/startup_service.sh
USER appuser

# ------------------------------- requirements ----------------------------- #

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --user -r /app/requirements.txt

# ------------------------------- ports ------------------------------------ #

EXPOSE 8000

# --------------------------- health check --------------------------------- #

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ping || exit 1

# ------------------------------- entry point ----------------------------- #

CMD ["/app/scripts/startup_service.sh"]