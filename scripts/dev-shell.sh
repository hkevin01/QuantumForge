#!/bin/bash
# Enter the development shell

set -e

# Check if GPU container is running
if docker-compose ps quantumforge-dev | grep -q "Up"; then
    echo "🚀 Entering GPU-enabled development environment..."
    docker-compose exec quantumforge-dev bash
elif docker-compose ps quantumforge-cpu | grep -q "Up"; then
    echo "🚀 Entering CPU-only development environment..."
    docker-compose exec quantumforge-cpu bash
else
    echo "❌ No development container is running."
    echo "Run './scripts/setup-dev.sh' first to start the development environment."
    exit 1
fi
