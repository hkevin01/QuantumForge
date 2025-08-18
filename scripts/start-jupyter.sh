#!/bin/bash
# Start Jupyter Lab server

set -e

echo "🔬 Starting Jupyter Lab server..."

# Start jupyter service if not already running
if ! docker-compose ps jupyter | grep -q "Up"; then
    echo "Starting Jupyter container..."
    docker-compose up -d jupyter
    echo "⏳ Waiting for Jupyter to be ready..."
    sleep 10
fi

echo ""
echo "🎉 Jupyter Lab is ready!"
echo ""
echo "🔗 Open in browser: http://localhost:8890"
echo "📁 Workspace: /workspace"
echo "📊 Examples: /workspace/examples/"
echo ""
echo "💡 Tips:"
echo "   • Create new notebooks in /workspace/notebooks/"
echo "   • Use 'import sys; sys.path.append(\"/workspace/src\")' to import quantumforge"
echo "   • GPU support is available if CUDA container is used"
echo ""

# Show container logs
echo "📋 Container logs:"
docker-compose logs --tail=20 jupyter
