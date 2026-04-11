ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
# Newer llama.cpp revisions add Gemma 4 support, but the floating prebuilt
# CUDA images failed to initialize on the RTX 4060 node with driver 550 /
# CUDA 12.4. Build llama.cpp from source against CUDA 12.4 instead so we keep
# GPU compatibility and regain newer model-architecture support.
ARG LLAMA_CPP_REF=3fc65063d9c356510b86fc2f15ca8aea711bfc47

FROM ${CUDA_DEVEL_IMAGE} AS llama-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src/llama.cpp

RUN git clone https://github.com/ggml-org/llama.cpp.git . && \
    git checkout "${LLAMA_CPP_REF}" && \
    cmake -S . -B build \
      -DGGML_CUDA=ON \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DBUILD_SHARED_LIBS=ON && \
    cmake --build build --target llama-server -j"$(nproc)" && \
    mkdir -p /tmp/llama-dist && \
    cp build/bin/llama-server /tmp/llama-dist/ && \
    find build \( -type f -o -type l \) \( -name 'libllama.so*' -o -name 'libggml*.so*' \) -exec cp -P {} /tmp/llama-dist/ \;

FROM ${CUDA_RUNTIME_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"

# Install Python tooling and isolate wrapper dependencies in a venv.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv curl libgomp1 && \
    python3 -m venv "${VIRTUAL_ENV}" && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=llama-builder /tmp/llama-dist/ /app/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# The wrapper manages llama-server internally and listens on 8080
EXPOSE 8080

# Clear the base image entrypoint so we control the process
ENTRYPOINT []
CMD ["python", "main.py"]
