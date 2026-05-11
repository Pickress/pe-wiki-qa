"""
app.py — Wiki Q&A chat app (Streamlit + ChromaDB + Claude)
Run: streamlit run app.py
"""

import os
import pathlib
import chromadb
import anthropic
import streamlit as st

CHROMA_PATH = "./chroma_db"
N_RESULTS = 10
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an expert analyst specializing in PE (polyethylene) resin for flexible packaging.
You answer questions using notes from an internal intelligence wiki covering:
- PE resin technology and grades (LLDPE, HDPE, mLLDPE, BOCD, MDO-PE, BOPE)
- Packaging regulations (PPWR, EPR, RecyClass, APR)
- Patents and IP landscape (FTO risk, assignees, claim scope)
- Competitor and market intelligence (Dow, ExxonMobil, Borealis, Sabic, LyondellBasell, SCGC)
- Sustainability, recyclability, PCR, carbon footprint trends

Instructions:
- Answer in Thai. Use technical English terms where appropriate (MDO-PE, PCR, FTO, PPWR, etc.)
- Synthesize information across multiple notes — do not just repeat one note
- Structure your answer with clear sections when the topic is complex
- Cite specific source notes at the end using their titles
- If the notes do not contain enough information, explicitly say what is missing and why — do not invent facts
- If asked for comparison, present it as a table"""

TRANSLATE_PROMPT = """You are a search query optimizer for a PE resin and flexible packaging intelligence wiki.
Convert the user's question (may be in Thai or English) into a short English search query (5-10 keywords) that will best retrieve relevant wiki notes.
Focus on TECHNICAL TERMS and TOPICS — not the question structure.
Examples:
  "MDOPE มีกี่ segment" → "MDO-PE market segments applications food packaging industrial"
  "ExxonMobil ทำอะไรบ้าง" → "ExxonMobil PE resin strategy patents BOCD mLLDPE"
  "PPWR deadline เมื่อไหร่" → "PPWR regulation deadline 2030 recyclability requirement"
Output ONLY the search keywords — no explanation, no preamble."""

STOP_WORDS = {"how","many","what","does","have","the","and","for","are","with","this","that","from","into"}

FOLDER_COLORS = {
    "patents": "#7c3aed",
    "mdope": "#7c3aed",
    "lamination": "#7c3aed",
    "library": "#7c3aed",
    "web-clips": "#0891b2",
    "themes": "#059669",
    "competitors": "#dc2626",
    "items": "#d97706",
    "weekly": "#d97706",
    "2026-W18": "#d97706",
    "2026-W19": "#d97706",
    "2026-W20": "#d97706",
    "ExxonMobil": "#dc2626",
    "Dow": "#dc2626",
    "sources": "#6b7280",
    "reports": "#6b7280",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Page background */
.stApp {
    background: #0f1117;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 820px; }

/* ── Header ── */
.wiki-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 24px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    margin-bottom: 24px;
}
.wiki-header .icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
}
.wiki-header h1 {
    font-size: 1.25rem; font-weight: 700;
    color: #f1f5f9; margin: 0; line-height: 1.2;
}
.wiki-header p {
    font-size: 0.8rem; color: #94a3b8; margin: 4px 0 0;
}
.wiki-header .badge {
    margin-left: auto; flex-shrink: 0;
    background: #1e3a5f; border: 1px solid #3b82f6;
    color: #93c5fd; font-size: 0.7rem; font-weight: 600;
    padding: 4px 10px; border-radius: 20px;
}

/* ── Chat messages ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* User bubble */
[data-testid="stChatMessageContent"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 14px 18px;
}

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, #0f2744 0%, #0f172a 100%);
    border: 1px solid #1d4ed8;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0 12px;
    color: #e2e8f0;
    line-height: 1.7;
}
.answer-card h2, .answer-card h3 {
    color: #93c5fd;
    font-size: 1rem;
    margin-top: 16px;
}
.answer-card table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin: 12px 0;
}
.answer-card th {
    background: #1e3a5f;
    color: #93c5fd;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #2563eb;
}
.answer-card td {
    padding: 8px 12px;
    border-bottom: 1px solid #1e293b;
    color: #cbd5e1;
}
.answer-card tr:hover td { background: #1e293b55; }

/* ── Query pill ── */
.query-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-bottom: 12px;
}
.query-pill span { color: #60a5fa; font-weight: 500; }

/* ── Source cards ── */
.sources-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 16px 0 8px;
}
.source-card {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
    text-decoration: none;
    transition: border-color 0.15s;
}
.source-card:hover { border-color: #475569; }
.source-card .num {
    font-size: 0.7rem;
    font-weight: 700;
    color: #475569;
    padding-top: 2px;
    min-width: 16px;
}
.source-card .info { flex: 1; min-width: 0; }
.source-card .title {
    font-size: 0.85rem;
    font-weight: 500;
    color: #cbd5e1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.source-card a .title { color: #60a5fa; }
.source-card .meta {
    display: flex; align-items: center; gap: 6px; margin-top: 3px;
}
.folder-badge {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 4px;
    border: 1px solid;
}

/* ── API key box ── */
.api-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    max-width: 480px;
    margin: 40px auto;
    text-align: center;
}
.api-box h3 { color: #f1f5f9; margin-bottom: 8px; }
.api-box p { color: #94a3b8; font-size: 0.85rem; margin-bottom: 20px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

/* Streamlit chat input */
[data-testid="stChatInput"] textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 2px #4f46e533 !important;
}
</style>
"""


def folder_badge_html(folder: str) -> str:
    color = FOLDER_COLORS.get(folder, "#6b7280")
    return f'<span class="folder-badge" style="color:{color};border-color:{color}33;background:{color}18">{folder}</span>'


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection("wiki_notes")


@st.cache_resource
def get_claude():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def translate_to_english(client, question: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=200, system=TRANSLATE_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text.strip()


def is_stub(doc: str) -> bool:
    s = doc.strip().lower()
    return len(s) < 100 or s.startswith("(see primary)")


def keyword_search(col, keywords: list[str], n: int = 20) -> list[dict]:
    all_results = col.get(limit=2000, include=["documents", "metadatas"])
    kw_lower = [k.lower() for k in keywords if len(k) > 3 and k.lower() not in STOP_WORDS]
    if not kw_lower:
        return []
    scored = []
    for doc_id, doc, meta in zip(all_results["ids"], all_results["documents"], all_results["metadatas"]):
        if is_stub(doc):
            continue
        doc_lower = doc.lower()
        hits = sum(1 for k in kw_lower if k in doc_lower)
        if hits > 0:
            scored.append({"id": doc_id, "text": doc, "meta": meta, "distance": 1 - hits/len(kw_lower), "kw_hits": hits})
    scored.sort(key=lambda x: x["kw_hits"], reverse=True)
    return scored[:n]


def search_wiki(col, query_en: str, query_original: str, n: int = N_RESULTS) -> list[dict]:
    vec_results = col.query(query_texts=[query_en], n_results=n * 2)
    vec_items = {}
    for i in range(len(vec_results["ids"][0])):
        doc_id = vec_results["ids"][0][i]
        doc = vec_results["documents"][0][i]
        if is_stub(doc):
            continue
        vec_items[doc_id] = {
            "id": doc_id, "text": doc,
            "meta": vec_results["metadatas"][0][i],
            "distance": vec_results["distances"][0][i],
        }
    keywords = [w for w in query_en.replace("-", " ").split() if len(w) > 3]
    kw_items = {item["id"]: item for item in keyword_search(col, keywords, n=n)}
    merged = dict(vec_items)
    for doc_id, item in kw_items.items():
        if doc_id not in merged:
            merged[doc_id] = item
    return list(merged.values())[:n]


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        title = meta.get("title", "")
        folder = meta.get("folder", "")
        url = meta.get("url", "")
        header = f"[Note {i}] {title} (folder: {folder})"
        if url:
            header += f"\nURL: {url}"
        parts.append(f"{header}\n{h['text'][:2000]}")
    return "\n\n---\n\n".join(parts)


def ask_claude(client, question: str, query_en: str, context: str) -> str:
    user_msg = f"""Wiki notes context (retrieved using query: "{query_en}"):

{context}

---
User question (Thai): {question}

Answer the question in Thai based on the context above. Synthesize across all relevant notes."""
    response = client.messages.create(
        model=MODEL, max_tokens=2048, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


def render_sources(hits: list[dict]) -> str:
    cards = []
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        title = meta.get("title", meta.get("file", "Untitled"))[:70]
        folder = meta.get("folder", "")
        url = meta.get("url", "")
        badge = folder_badge_html(folder)
        title_html = (
            f'<a href="{url}" target="_blank" style="text-decoration:none"><span class="title">{title}</span></a>'
            if url else f'<span class="title">{title}</span>'
        )
        cards.append(f"""
<div class="source-card">
  <span class="num">{i}</span>
  <div class="info">
    {title_html}
    <div class="meta">{badge}</div>
  </div>
</div>""")
    return "\n".join(cards)


# ── App ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="PE Wiki Q&A", page_icon="⬡", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⬡ PE Wiki Q&A")
    st.markdown("---")
    st.markdown("**Notes indexed**")
    if pathlib.Path(CHROMA_PATH).exists():
        try:
            _col = get_collection()
            st.markdown(f"`{_col.count():,}` notes")
        except Exception:
            st.markdown("`—`")
    st.markdown("**Model**")
    st.markdown(f"`{MODEL}`")
    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.markdown("<small style='color:#475569'>Refresh notes after adding new wiki entries: run `python ingest.py`</small>", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="wiki-header">
  <div class="icon">⬡</div>
  <div>
    <h1>PE Flexible Packaging — Wiki Q&A</h1>
    <p>ถามคำถามเกี่ยวกับ PE resin · regulations · patents · market intelligence</p>
  </div>
  <div class="badge">Internal</div>
</div>
""", unsafe_allow_html=True)

# DB check
if not pathlib.Path(CHROMA_PATH).exists():
    st.error("ยังไม่มีฐานข้อมูล — รัน `python ingest.py` ก่อน")
    st.stop()

# API key
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    st.markdown("""
    <div class="api-box">
      <h3>🔑 ใส่ API Key</h3>
      <p>ต้องการ Anthropic API key เพื่อเริ่มใช้งาน</p>
    </div>
    """, unsafe_allow_html=True)
    api_key = st.text_input("", type="password", placeholder="sk-ant-...", label_visibility="collapsed")
    if not api_key:
        st.stop()
    os.environ["ANTHROPIC_API_KEY"] = api_key
    st.rerun()

col = get_collection()
claude = get_claude()

if "history" not in st.session_state:
    st.session_state.history = []

# Render history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(f'<div class="answer-card">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources_html"):
                with st.expander(f"Sources — {msg['source_count']} notes"):
                    st.markdown(msg["sources_html"], unsafe_allow_html=True)
            if msg.get("query_en"):
                st.markdown(f'<div class="query-pill">🔍 Search: <span>{msg["query_en"]}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# Input
question = st.chat_input("ถามอะไรก็ได้ เช่น MDO-PE มี segment อะไรบ้าง?")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("กำลังวิเคราะห์คำถาม..."):
            query_en = translate_to_english(claude, question)

        with st.spinner(f"ค้นหาใน wiki..."):
            hits = search_wiki(col, query_en, question)
            context = build_context(hits)

        with st.spinner("สังเคราะห์คำตอบ..."):
            answer = ask_claude(claude, question, query_en, context)

        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

        sources_html = render_sources(hits)
        with st.expander(f"Sources — {len(hits)} notes"):
            st.markdown(sources_html, unsafe_allow_html=True)

        st.markdown(f'<div class="query-pill">🔍 Search: <span>{query_en}</span></div>', unsafe_allow_html=True)

    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "query_en": query_en,
        "sources_html": sources_html,
        "source_count": len(hits),
    })
