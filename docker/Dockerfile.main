# Multi-stage Dockerfile for QuantumForge development and production
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04 as base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    cmake \
    build-essential \
    git \
    curl \
    wget \
    libhdf5-dev \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libfftw3-dev \
    pkg-config \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for documentation tools
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set up working directory
WORKDIR /workspace

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install base Python packages
RUN pip install --upgrade pip setuptools wheel

# Development stage
FROM base as development

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    htop \
    tmux \
    fish \
    zsh \
    git-lfs \
    clang-format \
    valgrind \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Install Python development dependencies
COPY requirements.txt requirements-dev.txt /tmp/
RUN pip install -r /tmp/requirements-dev.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PySCF and quantum chemistry dependencies
RUN pip install pyscf h5py

# Copy source code
COPY . /workspace/

# Install QuantumForge in development mode (avoid extras; dev deps come from requirements-dev.txt)
RUN pip install -e "."

# Set up development environment
RUN echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> /root/.bashrc
RUN echo 'export CUDA_VISIBLE_DEVICES=0' >> /root/.bashrc

# Expose ports for Jupyter, Streamlit, and development servers
EXPOSE 8888 8501 8000 3000

# Default command for development
CMD ["bash"]

# Production stage
FROM base as production

# Copy only necessary files
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Install PyTorch with CUDA support (production optimized)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy source code
COPY src/ /workspace/src/
COPY pyproject.toml setup.cfg CMakeLists.txt /workspace/

# Install QuantumForge
RUN pip install .

# Create non-root user for security
RUN useradd -m -u 1000 quantumforge
USER quantumforge

# Production command
CMD ["python", "-m", "quantumforge.cli"]

# Testing stage
FROM development as testing

# Run tests
RUN pytest tests/ -v --cov=src/quantumforge --cov-report=html

# Documentation stage
FROM base as docs

# Install documentation dependencies
RUN pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints

# Copy source and docs
COPY src/ /workspace/src/
COPY docs/ /workspace/docs/
COPY pyproject.toml /workspace/

# Build documentation
WORKDIR /workspace/docs
RUN make html

# Serve documentation
EXPOSE 8080
CMD ["python", "-m", "http.server", "8080", "--directory", "_build/html"]
