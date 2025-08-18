# Multi-stage Dockerfile for QuantumForge Development
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-venv \
    git \
    cmake \
    build-essential \
    curl \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Development stage
FROM base as development

# Install development dependencies
RUN pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    matplotlib>=3.4.0 \
    jupyter \
    jupyterlab \
    pytest>=6.0.0 \
    pytest-cov \
    black \
    isort \
    mypy \
    flake8 \
    bandit \
    pre-commit \
    ipython \
    ipdb

WORKDIR /workspace

# Testing stage
FROM development as testing

# Copy test requirements
COPY requirements-test.txt* ./
RUN if [ -f requirements-test.txt ]; then pip install -r requirements-test.txt; fi

# Production stage
FROM base as production

# Install only production dependencies
RUN pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install \
    numpy>=1.21.0 \
    scipy>=1.7.0

WORKDIR /workspace

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package
RUN pip install -e .

CMD ["python"]
