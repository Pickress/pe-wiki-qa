#!/bin/zsh
# update_wiki.sh — Re-index Obsidian notes and push to Streamlit Cloud
# Run this whenever wiki notes are added or updated

cd "$(dirname "$0")"

echo "📚 Step 1/3 — Re-indexing wiki notes..."
.venv312/bin/python ingest.py

echo ""
echo "📦 Step 2/3 — Staging ChromaDB..."
git add chroma_db/

echo ""
echo "🚀 Step 3/3 — Pushing to GitHub..."
git commit -m "Update wiki index $(date '+%Y-%m-%d %H:%M')"
git push

echo ""
echo "✅ Done! Streamlit Cloud will redeploy in ~1-2 minutes."
echo "   App URL: https://pe-wiki-app-jylce4ueb44yakxkp24ddm.streamlit.app"
