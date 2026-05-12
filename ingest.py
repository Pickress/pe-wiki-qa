"""
ingest.py — Load Obsidian notes into ChromaDB for RAG
Run once (or re-run to refresh): python ingest.py
"""

import os
import re
import yaml
import hashlib
import chromadb
from pathlib import Path

VAULT_PATH = Path("/Users/macintosh/Obsidian/External-Intelligence")
CHROMA_PATH = Path("./chroma_db")

SKIP_DIRS = {"_templates", ".obsidian", ".trash"}
SKIP_FILES = {"dataview-queries.md"}


def parse_note(file_path: Path) -> list[dict] | None:
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    # Split frontmatter
    frontmatter = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                pass
            body = parts[2].strip()

    # Skip empty notes
    if len(body) < 50:
        return None

    # Clean wikilinks [[Target|Label]] → Label or Target
    body = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', body)
    body = re.sub(r'\[\[([^\]]+)\]\]', r'\1', body)

    # Build metadata (ChromaDB only accepts str/int/float/bool values)
    meta = {
        "file": str(file_path.relative_to(VAULT_PATH)),
        "folder": file_path.parent.name,
        "title": str(frontmatter.get("title", file_path.stem)),
        "type": str(frontmatter.get("type", "note")),
    }

    # Optional fields
    tags = frontmatter.get("tags", [])
    if isinstance(tags, list):
        meta["tags"] = ",".join(str(t) for t in tags)
    elif tags:
        meta["tags"] = str(tags)

    for field in ("assignee", "status", "fto-risk", "cluster", "pub-date"):
        val = frontmatter.get(field)
        if val is not None:
            meta[field] = str(val)

    url = frontmatter.get("url") or frontmatter.get("source_url")
    if url:
        meta["url"] = str(url)

    title = meta["title"]
    file_md5 = hashlib.md5(str(file_path).encode()).hexdigest()

    # Split at H2/H3 headers so each section gets its own embedding
    raw_parts = re.split(r'\n(#{2,3} [^\n]+)', body)
    sections = []
    if raw_parts[0].strip():
        sections.append(raw_parts[0])
    for i in range(1, len(raw_parts), 2):
        header = raw_parts[i]
        content = raw_parts[i + 1] if i + 1 < len(raw_parts) else ""
        sections.append(f"{header}\n{content}")

    chunks = []
    for idx, section in enumerate(sections):
        chunk_body = section.strip()
        if len(chunk_body) < 30:
            continue
        chunk_text = f"{title}\n\n{chunk_body}"[:3000]
        chunks.append({"id": f"{file_md5}_{idx}", "text": chunk_text, "meta": meta})

    return chunks if chunks else None


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # Delete and recreate collection so old single-chunk IDs are removed
    try:
        client.delete_collection("wiki_notes")
    except Exception:
        pass
    col = client.create_collection("wiki_notes")

    md_files = [
        p for p in VAULT_PATH.rglob("*.md")
        if not any(skip in p.parts for skip in SKIP_DIRS)
        and p.name not in SKIP_FILES
    ]

    print(f"Found {len(md_files)} notes — ingesting...")

    ids, docs, metas = [], [], []
    skipped = 0
    for f in md_files:
        chunks = parse_note(f)
        if chunks is None:
            skipped += 1
            continue
        for chunk in chunks:
            ids.append(chunk["id"])
            docs.append(chunk["text"])
            metas.append(chunk["meta"])

    # Upsert in batches of 100
    batch = 100
    for i in range(0, len(ids), batch):
        col.upsert(
            ids=ids[i:i+batch],
            documents=docs[i:i+batch],
            metadatas=metas[i:i+batch],
        )
        print(f"  Upserted {min(i+batch, len(ids))}/{len(ids)}")

    print(f"Done. {len(ids)} chunks from {len(md_files) - skipped} notes ({skipped} skipped).")
    print(f"ChromaDB stored at: {CHROMA_PATH.resolve()}")


if __name__ == "__main__":
    main()
