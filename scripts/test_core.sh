#!/bin/bash
"""
Quick test script to run the core demo in our Docker environment
"""

# Navigate to project root
cd /home/kevin/Projects/QuantumForge

# Build and run the development container
echo "Building QuantumForge development environment..."
docker-compose -f docker/docker-compose.yml build dev

echo "Running core module integration test..."
docker-compose -f docker/docker-compose.yml run --rm dev python examples/core_demo.py

echo "Test completed!"
