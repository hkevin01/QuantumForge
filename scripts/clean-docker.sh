#!/bin/bash
# Clean Docker environment

set -e

echo "🧹 Cleaning QuantumForge Docker environment..."

# Stop all containers
echo "⏹️  Stopping all containers..."
docker-compose down

# Remove containers
echo "🗑️  Removing containers..."
docker-compose rm -f

# Remove images (optional)
read -p "🔥 Remove Docker images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing QuantumForge images..."
    docker images | grep quantumforge | awk '{print $3}' | xargs docker rmi -f 2>/dev/null || true
fi

# Remove volumes (optional)
read -p "💥 Remove persistent volumes? This will delete all data! (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing volumes..."
    docker-compose down -v
    docker volume prune -f
fi

# Clean up build cache
echo "🧽 Cleaning build cache..."
docker builder prune -f

echo ""
echo "✅ Docker environment cleaned!"
echo ""
echo "🔄 To restart development environment:"
echo "   ./scripts/setup-dev.sh"
