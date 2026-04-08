FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Install Python 3 and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py .

# The wrapper manages llama-server internally and listens on 8080
EXPOSE 8080

# Clear the base image entrypoint so we control the process
ENTRYPOINT []
CMD ["python3", "main.py"]
