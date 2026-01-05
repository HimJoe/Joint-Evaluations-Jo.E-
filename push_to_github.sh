#!/bin/bash

# Jo.E GitHub Push Script
# This script helps you push the code to GitHub

echo "🚀 Jo.E GitHub Push Helper"
echo "=========================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the project directory."
    exit 1
fi

# Display current status
echo "📊 Current Git Status:"
git status --short
echo ""

# Show what will be pushed
echo "📦 Commits to push:"
git log origin/main..HEAD --oneline 2>/dev/null || git log --oneline -3
echo ""

# Configure git to use credential helper
echo "🔧 Configuring Git credential helper..."
git config --global credential.helper osxkeychain

# Reset remote URL to HTTPS (without token embedded)
echo "🔗 Setting up remote URL..."
git remote set-url origin https://github.com/HimJoe/Joint-Evaluations-Jo.E-.git

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Run: git push -u origin main"
echo "2. When prompted for username, enter: HimJoe"
echo "3. When prompted for password, paste your Personal Access Token"
echo ""
echo "💡 Tip: The token will be saved to macOS Keychain for future use"
echo ""

# Optional: Try to push automatically
read -p "🤔 Do you want to try pushing now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Attempting to push..."
    git push -u origin main
fi
