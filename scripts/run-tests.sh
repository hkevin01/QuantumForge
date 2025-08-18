#!/bin/bash
# Run comprehensive test suite

set -e

echo "🧪 Running QuantumForge test suite..."

# Function to run tests in a container
run_tests() {
    local container=$1
    echo "Running tests in $container..."

    docker-compose exec $container bash -c "
        cd /workspace &&
        echo '📦 Installing test dependencies...' &&
        pip install -e '.[dev]' &&
        echo '🔍 Running linting checks...' &&
        flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503 &&
        echo '🎯 Running type checks...' &&
        mypy src/quantumforge --ignore-missing-imports &&
        echo '🧪 Running unit tests...' &&
        pytest tests/ -v --cov=src/quantumforge --cov-report=xml --cov-report=html &&
        echo '🔒 Running security checks...' &&
        bandit -r src/ -f json -o security-report.json || true &&
        echo '✅ All tests completed!'
    "
}

# Check which container is available and run tests
if docker-compose ps quantumforge-dev | grep -q "Up"; then
    run_tests quantumforge-dev
elif docker-compose ps quantumforge-cpu | grep -q "Up"; then
    run_tests quantumforge-cpu
else
    echo "❌ No development container is running."
    echo "Starting CPU container for testing..."
    docker-compose up -d quantumforge-cpu
    sleep 5
    run_tests quantumforge-cpu
fi

# Copy test results to host
echo "📊 Copying test results to host..."
mkdir -p test-results
docker-compose cp quantumforge-dev:/workspace/htmlcov ./test-results/ 2>/dev/null || \
docker-compose cp quantumforge-cpu:/workspace/htmlcov ./test-results/ 2>/dev/null || true

docker-compose cp quantumforge-dev:/workspace/coverage.xml ./test-results/ 2>/dev/null || \
docker-compose cp quantumforge-cpu:/workspace/coverage.xml ./test-results/ 2>/dev/null || true

docker-compose cp quantumforge-dev:/workspace/security-report.json ./test-results/ 2>/dev/null || \
docker-compose cp quantumforge-cpu:/workspace/security-report.json ./test-results/ 2>/dev/null || true

echo "📈 Test results available in ./test-results/"
echo "🌐 Open ./test-results/htmlcov/index.html to view coverage report"
