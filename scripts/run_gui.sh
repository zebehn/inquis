#!/bin/bash
# Launch the Streamlit GUI for Visual Perception Agent

cd "$(dirname "$0")/.."

echo "========================================"
echo "Visual Perception Agent - GUI"
echo "========================================"
echo ""
echo "Launching Streamlit application..."
echo "The app will open in your default browser."
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run src/gui/app.py \
  --server.port 8501 \
  --server.address localhost \
  --server.headless false \
  --browser.gatherUsageStats false
