ARG CUDA_VERSION=12.6.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git curl sudo ffmpeg fonts-noto wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Clean apt cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Workaround for CUDA compatibility issues
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# Set working directory
WORKDIR /app

# Copy project files (build from local context, not clone)
COPY . .

# Install PyTorch with CUDA 12.6
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install project dependencies
RUN pip install --no-cache-dir -e .

# Set CUDA-related environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set CUDA architecture list
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

EXPOSE 8501

CMD ["streamlit", "run", "st.py"]
