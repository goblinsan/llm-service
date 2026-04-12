ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
# Newer llama.cpp revisions add Gemma 4 support, but the floating prebuilt
# CUDA images failed to initialize on the RTX 4060 node with driver 550 /
# CUDA 12.4. Build llama.cpp from source against CUDA 12.4 instead so we keep
# GPU compatibility and regain newer model-architecture support.
ARG LLAMA_CPP_REF=master
ARG LLAMA_CUDA_ARCHITECTURES=89

FROM ${CUDA_DEVEL_IMAGE} AS llama-builder

ENV DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_REF
ARG LLAMA_CUDA_ARCHITECTURES

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src/llama.cpp

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}" && \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" && \
    curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/refs/heads/${LLAMA_CPP_REF}.tar.gz" \
      | tar -xz --strip-components=1 && \
    cmake -S . -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${LLAMA_CUDA_ARCHITECTURES} \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DBUILD_SHARED_LIBS=ON && \
    cmake --build build --target llama-server -j"$(nproc)" && \
    mkdir -p /tmp/llama-dist && \
    LLAMA_SERVER_BIN="$(find build -type f -name llama-server | head -n 1)" && \
    test -n "$LLAMA_SERVER_BIN" && \
    install -m 0755 "$LLAMA_SERVER_BIN" /tmp/llama-dist/llama-server && \
    find build \( -type f -o -type l \) -name 'lib*.so*' -exec cp -P {} /tmp/llama-dist/ \;

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
