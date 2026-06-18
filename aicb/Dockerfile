# Base image: Official NVIDIA PyTorch image with Python 3 and GPU support.
FROM nvcr.io/nvidia/pytorch:25.05-py3

# Install git for version control operations and clean up apt cache.
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set the application's working directory.
WORKDIR /workspace/AICB

# [Optional] Configure pip and uv to use Aliyun mirror for faster package downloads.
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com
ENV UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple"

RUN pip install --no-cache-dir uv

# Copy only the requirements file first to leverage Docker's layer cache.
# This layer is rebuilt only when requirements.txt changes.
COPY requirements.txt .

# Install Python dependencies using uv.
RUN UV_TORCH_BACKEND=auto uv pip install -v --system --no-cache-dir --no-build-isolation --break-system-packages -r requirements.txt

# Copy the rest of the application source code into the image.
COPY . .

RUN mv ./workload_generator /usr/local/lib/python3.12/dist-packages &&\
    mv ./utils /usr/local/lib/python3.12/dist-packages &&\
    mv ./log_analyzer /usr/local/lib/python3.12/dist-packages
ENV PATH="/workspace/AICB/.venv/bin:$PATH"