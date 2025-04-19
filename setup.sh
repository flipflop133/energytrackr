#!/usr/bin/env bash

set -e

echo "🛠  Running initial setup via Makefile..."
make install

echo ""
echo "📥 Activating virtual environment..."
echo "💡 Tip: To deactivate, type 'deactivate'"
echo ""

# Activate the virtual environment and drop into a shell
source .venv/bin/activate
exec "$SHELL"

echo "Now you can run the app with:"
echo "energytrackr"
