# Ajuste as versões conforme sua GPU/driver. TensorRT 10.x + CUDA 12.x + cuDNN 9.
FROM nvcr.io/nvidia/tensorrt:24.08-py3

ARG ORT_VERSION=1.19.2
RUN apt-get update && apt-get install -y --no-install-recommends cmake build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime GPU (C++)
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz \
    && tar xzf onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz -C /opt \
    && mv /opt/onnxruntime-linux-x64-gpu-${ORT_VERSION} /opt/onnxruntime
ENV ORT_ROOT=/opt/onnxruntime LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN cmake -B build -DORT_ROOT=${ORT_ROOT} && cmake --build build -j
CMD ["bash"]
