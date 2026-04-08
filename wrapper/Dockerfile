FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Install Python tooling and isolate wrapper dependencies in a venv so
# Ubuntu 24.04's externally managed Python policy does not break the build.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv curl && \
    python3 -m venv "${VIRTUAL_ENV}" && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# The wrapper manages llama-server internally and listens on 8080
EXPOSE 8080

# Clear the base image entrypoint so we control the process
ENTRYPOINT []
CMD ["python", "main.py"]
