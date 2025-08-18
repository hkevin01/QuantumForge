#!/bin/bash
# Development environment setup script

set -e

echo "🚀 Setting up QuantumForge development environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check for NVIDIA Docker support
if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
    echo "✅ NVIDIA Docker support detected"
    USE_GPU=true
else
    echo "⚠️  No NVIDIA Docker support detected. Using CPU-only mode."
    USE_GPU=false
fi

# Create necessary directories
mkdir -p data/{raw,processed,models,results}
mkdir -p logs
mkdir -p .cache

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# QuantumForge Environment Configuration
PYTHONPATH=/workspace/src
JUPYTER_ENABLE_LAB=yes
CUDA_VISIBLE_DEVICES=0

# Database Configuration
POSTGRES_DB=quantumforge
POSTGRES_USER=quantumforge
POSTGRES_PASSWORD=quantumforge123

# MinIO Configuration
MINIO_ROOT_USER=quantumforge
MINIO_ROOT_PASSWORD=quantumforge123

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_ACCESS_KEY_ID=quantumforge
AWS_SECRET_ACCESS_KEY=quantumforge123
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
EOF
fi

# Build and start services
echo "🔨 Building Docker images..."
if [ "$USE_GPU" = true ]; then
    docker-compose build quantumforge-dev
    echo "🚀 Starting GPU-enabled development environment..."
    docker-compose up -d quantumforge-dev postgres redis minio mlflow
else
    docker-compose build quantumforge-cpu
    echo "🚀 Starting CPU-only development environment..."
    docker-compose up -d quantumforge-cpu postgres redis minio mlflow
fi

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Setup MinIO buckets
echo "📦 Setting up MinIO buckets..."
docker-compose exec minio mc alias set local http://localhost:9000 quantumforge quantumforge123
docker-compose exec minio mc mb local/mlflow || true
docker-compose exec minio mc mb local/data || true

# Display service URLs
echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "📊 Service URLs:"
echo "   • Main Development: docker-compose exec quantumforge-dev bash"
if [ "$USE_GPU" = false ]; then
    echo "   • CPU Development:  docker-compose exec quantumforge-cpu bash"
fi
echo "   • Jupyter Lab:      http://localhost:8890"
echo "   • Streamlit App:    http://localhost:8503"
echo "   • MLflow:           http://localhost:5000"
echo "   • MinIO Console:    http://localhost:9001"
echo "   • Documentation:    http://localhost:8080"
echo "   • SonarQube:        http://localhost:9002"
echo ""
echo "🔧 Development Commands:"
echo "   • Enter dev env:    ./scripts/dev-shell.sh"
echo "   • Run tests:        ./scripts/run-tests.sh"
echo "   • Start Jupyter:    ./scripts/start-jupyter.sh"
echo "   • Start Streamlit:  ./scripts/start-streamlit.sh"
echo "   • Clean all:        ./scripts/clean-docker.sh"
echo ""
echo "📚 Next steps:"
echo "   1. Run: ./scripts/dev-shell.sh"
echo "   2. Inside container: pytest tests/"
echo "   3. Open Jupyter: http://localhost:8890"
echo ""
