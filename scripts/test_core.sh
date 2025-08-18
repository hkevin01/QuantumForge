#!/bin/bash
#
# Quick test script to run the core demo in our Docker environment
#

# Navigate to project root
cd /home/kevin/Projects/QuantumForge

# Build and run the development container (CPU version for testing)
echo "Building QuantumForge development environment (CPU)..."
docker-compose -f docker-compose.yml -f docker-compose.test.yml build quantumforge-dev

echo "Running core module integration test..."
docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm quantumforge-dev python examples/core_demo.py

echo "Test completed!"
