# Base image: Official NVIDIA PyTorch image with Python 3 and GPU support.
FROM nvcr.io/nvidia/pytorch:25.05-py3

# Install git for version control operations and clean up apt cache.
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set the application's working directory.
WORKDIR /workspace/SimAI

# [Optional] Configure pip and uv to use Aliyun mirror for faster package downloads.
# Use HTTPS to avoid MITM during dependency download (no trusted-host needed with TLS).
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
ENV UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple"

RUN pip install --no-cache-dir uv

# Copy only the requirements file first to leverage Docker's layer cache.
# This layer is rebuilt only when requirements.txt changes.
COPY aicb/requirements.txt /tmp/reqs/aicb.txt
COPY vidur-alibabacloud/requirements.txt /tmp/reqs/vidur.txt

# Install Python dependencies using uv.
RUN UV_TORCH_BACKEND=auto uv pip install -v --system --no-cache-dir --no-build-isolation --break-system-packages -r /tmp/reqs/aicb.txt &&\
    UV_TORCH_BACKEND=auto uv pip install -v --system --no-cache-dir --no-build-isolation --break-system-packages -r /tmp/reqs/vidur.txt

# Copy the rest of the application source code into the image.
COPY . .

# Move helper packages onto the Python site-packages path. Compute the path
# dynamically so this does not break if the base image's Python minor version
# or site-packages location changes.
RUN SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')" &&\
    mv ./workload_generator "$SITE_PACKAGES" &&\
    mv ./utils "$SITE_PACKAGES" &&\
    mv ./log_analyzer "$SITE_PACKAGES"
ENV PYTHONPATH=/workspace/SimAI:/workspace/SimAI/aicb:/workspace/SimAI/vidur:$PYTHONPATH