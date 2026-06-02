#!/bin/zsh
# update_wiki.sh — Re-index Obsidian notes and push to Streamlit Cloud
# Run this whenever wiki notes are added or updated

cd "$(dirname "$0")"

GRAPH_SRC="/Users/macintosh/Obsidian/External-Intelligence/.understand-anything/knowledge-graph.json"
GRAPH_DST="./knowledge-graph.json"

echo "📚 Step 1/4 — Re-indexing wiki notes..."
.venv312/bin/python ingest.py

echo ""
echo "🧠 Step 2/4 — Syncing knowledge graph..."
if [ -f "$GRAPH_SRC" ]; then
  cp "$GRAPH_SRC" "$GRAPH_DST"
  echo "   Copied knowledge-graph.json ($(du -h $GRAPH_DST | cut -f1))"
else
  echo "   ⚠️  knowledge-graph.json not found — skipping graph sync"
  echo "   Run /understand-knowledge to regenerate"
fi

echo ""
echo "📦 Step 3/4 — Staging files..."
git add chroma_db/ knowledge-graph.json

echo ""
echo "🚀 Step 4/4 — Pushing to GitHub..."
git commit -m "Update wiki index + knowledge graph $(date '+%Y-%m-%d %H:%M')"
git push

echo ""
echo "✅ Done! Streamlit Cloud will redeploy in ~1-2 minutes."
echo "   App URL: https://pe-wiki-app-jylce4ueb44yakxkp24ddm.streamlit.app"
