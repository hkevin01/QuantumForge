#!/bin/bash
# Start Streamlit application

set -e

echo "🎨 Starting Streamlit application..."

# Start streamlit service if not already running
if ! docker-compose ps streamlit | grep -q "Up"; then
    echo "Starting Streamlit container..."
    docker-compose up -d streamlit
    echo "⏳ Waiting for Streamlit to be ready..."
    sleep 10
fi

echo ""
echo "🎉 Streamlit application is ready!"
echo ""
echo "🔗 Open in browser: http://localhost:8503"
echo "🎨 App source: /workspace/src/quantumforge/gui/app_streamlit.py"
echo ""
echo "💡 Tips:"
echo "   • The app auto-reloads when you modify the source code"
echo "   • Check container logs for debugging: docker-compose logs streamlit"
echo "   • Access development tools through the sidebar"
echo ""

# Show container logs
echo "📋 Container logs:"
docker-compose logs --tail=20 streamlit
