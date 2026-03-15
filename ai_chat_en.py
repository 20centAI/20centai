#!/usr/bin/env python3
# ai_chat_en.py v6.2 - Single AI Chat with Long-Term Memory
# Choose your AI partner: Claude, GPT-4o, Mistral, DeepSeek, Perplexity, Qwen, Groq
# Features: 20-Message-Compression (90% token savings), FTS5 Archive, Cost Tracking
# 
# The code was created by claude.ai in cooperation with perplexity.ai and deepseek.com
# It is free for everyone and released under the MIT License. Have fun with it.

import streamlit as st
import anthropic
import requests
import re
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ========== CONFIG ==========
st.set_page_config(page_title="20centAI", page_icon="🏛️", layout="wide")

CHAT_FILE = "council_chat.md"
DB_PATH   = "ai_council.db"
COMPRESS_THRESHOLD = 40  # start compression after N messages
KEEP_ORIGINAL      = 20  # keep last N messages verbatim (the 'gold standard')

# url=None  → Anthropic SDK (special auth)
# url=str   → OpenAI-compatible REST endpoint (all others)
MODELS = {
    # ── Anthropic ──────────────────────────────────────────────────────────────
    "Claude Sonnet 4.5":  {"id": "claude-sonnet-4-6",          "env": "ANTHROPIC_API_KEY",  "url": None},
    "Claude Haiku 4.5":   {"id": "claude-haiku-4-5-20251001",  "env": "ANTHROPIC_API_KEY",  "url": None},

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    "GPT-4o":             {"id": "gpt-4o",                     "env": "OPENAI_API_KEY",     "url": "https://api.openai.com/v1/chat/completions"},
    "GPT-4o mini":        {"id": "gpt-4o-mini",                "env": "OPENAI_API_KEY",     "url": "https://api.openai.com/v1/chat/completions"},

    # ── Mistral ────────────────────────────────────────────────────────────────
    "Mistral Large":      {"id": "mistral-large-latest",       "env": "MISTRAL_API_KEY",    "url": "https://api.mistral.ai/v1/chat/completions"},
    "Mistral Small":      {"id": "mistral-small-latest",       "env": "MISTRAL_API_KEY",    "url": "https://api.mistral.ai/v1/chat/completions"},

    # ── DeepSeek ───────────────────────────────────────────────────────────────
    "DeepSeek V3":        {"id": "deepseek-chat",              "env": "DEEPSEEK_API_KEY",   "url": "https://api.deepseek.com/chat/completions"},
    "DeepSeek R1":        {"id": "deepseek-reasoner",          "env": "DEEPSEEK_API_KEY",   "url": "https://api.deepseek.com/chat/completions"},

    # ── Perplexity ─────────────────────────────────────────────────────────────
    "Perplexity Sonar":   {"id": "sonar-pro",                  "env": "PERPLEXITY_API_KEY", "url": "https://api.perplexity.ai/chat/completions"},

    # ── Qwen (Alibaba) ─────────────────────────────────────────────────────────
    "Qwen Max":           {"id": "qwen-max",                   "env": "DASHSCOPE_API_KEY",  "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"},
    "Qwen Plus":          {"id": "qwen-plus",                  "env": "DASHSCOPE_API_KEY",  "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"},
    "Qwen Turbo":         {"id": "qwen-turbo",                 "env": "DASHSCOPE_API_KEY",  "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"},

    # ── Groq (fast & cheap inference) ─────────────────────────────────────────
    "Groq Llama 3.3 70B": {"id": "llama-3.3-70b-versatile",   "env": "GROQ_API_KEY",       "url": "https://api.groq.com/openai/v1/chat/completions"},
    "Groq Llama 3.1 8B":  {"id": "llama-3.1-8b-instant",      "env": "GROQ_API_KEY",       "url": "https://api.groq.com/openai/v1/chat/completions"},
}

# Prices in USD per 1M tokens (input / output)
# Last verified: March 2026 — check provider docs for updates
PRICES = {
    "Claude Sonnet 4.5":  {"input": 3.0,   "output": 15.0},
    "Claude Haiku 4.5":   {"input": 1.0,   "output": 5.0},
    "GPT-4o":             {"input": 2.5,   "output": 10.0},
    "GPT-4o mini":        {"input": 0.15,  "output": 0.6},
    "Mistral Large":      {"input": 0.5,   "output": 1.5},
    "Mistral Small":      {"input": 0.1,   "output": 0.3},
    "DeepSeek V3":        {"input": 0.0281,  "output": 0.42},
    "DeepSeek R1":        {"input": 0.0281,  "output": 0.42},
    "Perplexity Sonar":   {"input": 1.0,   "output": 1.0},
    "Qwen Max":           {"input": p.3,   "output": 1.49},
    "Qwen Plus":          {"input": 0.2,   "output": 0.8},
    "Qwen Turbo":         {"input": 0.1,  "output": 0.4},
    "Groq Llama 3.3 70B": {"input": 0.59,  "output": 0.79},
    "Groq Llama 3.1 8B":  {"input": 0.05,  "output": 0.08},
}

# All known model names for the chat renderer
ALL_MODEL_NAMES = list(MODELS.keys())

def get_key(name: str) -> str:
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""

def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES.get(model, {"input": 0, "output": 0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


# ========== SQLITE ARCHIVE ==========
class Archive:
    """
    Flat schema: one messages table + FTS5 full-text search.
    WAL mode for concurrency. Messages are immutable (append-only).
    """
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                author  TEXT NOT NULL,
                content TEXT NOT NULL,
                ts      TEXT NOT NULL,
                topic   TEXT DEFAULT 'current'
            )
        """)
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
            USING fts5(content, author, topic, content=messages, content_rowid=id)
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ts    ON messages(ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_topic ON messages(topic)")
        self.conn.commit()

    def save_message(self, author: str, content: str, topic: str = "current"):
        cur = self.conn.cursor()
        ts = datetime.now().isoformat()
        cur.execute(
            "INSERT INTO messages (author, content, ts, topic) VALUES (?, ?, ?, ?)",
            (author, content, ts, topic)
        )
        cur.execute(
            "INSERT INTO messages_fts (rowid, content, author, topic) VALUES (?, ?, ?, ?)",
            (cur.lastrowid, content, author, topic)
        )
        self.conn.commit()

    def archive_topic(self, topic: str, messages: list):
        """Save all messages from a session under a topic tag."""
        cur = self.conn.cursor()
        for m in messages:
            cur.execute(
                "INSERT INTO messages (author, content, ts, topic) VALUES (?, ?, ?, ?)",
                (m["author"], m["content"], m.get("ts", datetime.now().isoformat()), topic)
            )
            cur.execute(
                "INSERT INTO messages_fts (rowid, content, author, topic) VALUES (?, ?, ?, ?)",
                (cur.lastrowid, m["content"], m["author"], topic)
            )
        self.conn.commit()

    def search(self, query: str, limit: int = 3) -> str:
        """FTS5 full-text search — returns a formatted context string for prompt injection."""
        try:
            query_clean = re.sub(r'[^\w\s]', ' ', query).strip()
            if not query_clean:
                return ""
            cur = self.conn.cursor()
            cur.execute("""
                SELECT m.author, m.content, m.ts, m.topic
                FROM messages_fts fts
                JOIN messages m ON fts.rowid = m.id
                WHERE messages_fts MATCH ?
                ORDER BY rank LIMIT ?
            """, (query_clean, limit))
            hits = cur.fetchall()
            if not hits:
                return ""
            result = f"\n\n---\n📚 ARCHIVE RESULTS for '{query}':\n"
            for author, content, ts, topic in hits:
                result += f"[{topic} | {author} | {ts[:10]}]: {content[:120]}...\n"
            result += "---\n"
            return result
        except Exception:
            return ""

    def list_topics(self) -> list:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT topic, COUNT(*) as count, MAX(ts) as last
            FROM messages
            WHERE topic != 'current'
            GROUP BY topic ORDER BY last DESC
        """)
        return cur.fetchall()

    def stats(self) -> dict:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM messages")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT topic) FROM messages")
        topics = cur.fetchone()[0]
        return {"total": total, "topics": topics}


# ========== CHAT FILE HELPERS ==========
def load_chat() -> str:
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return f"# 20centAI Session\n*Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n---\n\n"

def save_chat(content: str):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

def count_messages(chat: str) -> int:
    # Count USER messages + any known AI model name messages
    model_pattern = "|".join(re.escape(m) for m in ALL_MODEL_NAMES)
    return len(re.findall(
        rf'\[(\d{{2}}:\d{{2}}) \| (?:USER|{model_pattern})\]',
        chat
    ))

def append_message(chat: str, role: str, text: str, escape_html: bool = False) -> str:
    # escape_html=True for user input only — AI responses must render Markdown as-is
    safe_text = text.replace("<", "&lt;").replace(">", "&gt;") if escape_html else text
    ts = datetime.now().strftime("%H:%M")
    return chat + f"\n[{ts} | {role}]: {safe_text}\n"

def extract_messages(chat: str) -> list:
    # Match USER and any model name as role
    model_pattern = "|".join(re.escape(m) for m in ALL_MODEL_NAMES)
    messages = []
    pattern = rf'\[(\d{{2}}:\d{{2}}) \| (USER|{model_pattern})\]: (.*?)(?=\n\[|\Z)'
    for match in re.finditer(pattern, chat, re.DOTALL):
        messages.append({
            "ts":      match.group(1),
            "author":  match.group(2),
            "content": match.group(3).strip()
        })
    return messages


# ========== SUMMARIZATION HELPER ==========
def summarize_text(text: str, prompt: str, model_name: str) -> str:
    """
    Calls the active model for a short summary (compression & archiving).
    Same two-branch design as ai_response() — no per-model boilerplate.
    """
    cfg = MODELS[model_name]
    key = get_key(cfg["env"])
    if not key:
        return ""
    sys_prompt = "You are a concise summarizer. Reply with the summary only."
    try:
        if cfg["url"] is None:
            client = anthropic.Anthropic(api_key=key)
            resp = client.messages.create(
                model=cfg["id"], max_tokens=400,
                system=sys_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
        else:
            resp = requests.post(
                cfg["url"],
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": cfg["id"],
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": prompt}
                    ],
                    "max_tokens": 400,
                }, timeout=25
            )
            return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""


# ========== TOPIC ARCHIVING ==========
def archive_and_reset(topic: str, archive: Archive, model_name: str) -> str:
    """Archive current chat to SQLite, summarize via active model, then start fresh."""
    try:
        chat = load_chat()
        messages = extract_messages(chat)
        if not messages:
            return "No messages to archive."

        archive.archive_topic(topic, messages)

        text = "\n".join([f"[{m['author']}]: {m['content'][:200]}" for m in messages[:25]])
        prompt = f"Write a concise archive summary (max 8 sentences).\nTopic: {topic}\n\nCHAT:\n{text}"
        summary = summarize_text(text, prompt, model_name)
        if summary:
            archive.save_message("SUMMARY", summary, topic)

        new_chat = (f"# 20centAI Session\n"
                    f"*Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
                    f"*Previous topic archived: {topic}*\n\n---\n\n")
        save_chat(new_chat)
        return f"✅ '{topic}' archived ({len(messages)} messages)"
    except Exception as e:
        return f"❌ Error: {e}"


# ========== COMPRESSION ==========
def compress_chat(chat: str, model_name: str) -> str:
    """
    Core feature: compress oldest messages into a summary, keep last KEEP_ORIGINAL verbatim.
    Always uses the active model — no hidden DeepSeek dependency.
    """
    try:
        messages = extract_messages(chat)
        if len(messages) < COMPRESS_THRESHOLD:
            return chat

        to_compress = messages[:-KEEP_ORIGINAL]
        to_keep     = messages[-KEEP_ORIGINAL:]

        text   = "\n".join([f"[{m['author']}]: {m['content']}" for m in to_compress])
        prompt = f"Summarize the following chat (max 8 sentences, key points only):\n{text}"
        summary = summarize_text(text, prompt, model_name)

        if not summary:
            st.warning("Compression skipped: no summary returned.")
            return chat

        header_match = re.match(r'(.*?)(?=\[\d{2}:\d{2} \|)', chat, re.DOTALL)
        header = header_match.group(1) if header_match else "# 20centAI\n\n---\n\n"

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_chat  = header
        new_chat += f"\n## 📋 SUMMARY ({model_name}, {ts})\n"
        new_chat += f"*{len(to_compress)} messages compressed · {KEEP_ORIGINAL} kept verbatim*\n\n"
        new_chat += summary
        new_chat += "\n\n---\n\n## 📝 RECENT MESSAGES\n\n"
        for m in to_keep:
            new_chat += f"\n[{m['ts']} | {m['author']}]: {m['content']}\n"

        return new_chat
    except Exception as e:
        st.warning(f"Compression failed: {e}")
        return chat


# ========== AI RESPONSE ==========
def ai_response(context: str, model_name: str, system_context: str = "") -> tuple:
    """
    Two-branch design:
      - Claude  → Anthropic SDK (different auth model)
      - Everyone else → single OpenAI-compatible REST call
    Adding a new model = add one entry to MODELS dict, zero code changes here.

    system_context: archive results injected into prompt only, never saved to chat.
    """
    cfg = MODELS[model_name]
    key = get_key(cfg["env"])
    if not key:
        return f"❌ {cfg['env']} not found in .env", 0, 0

    system_prompt = "You are a helpful AI assistant. Be precise and clear."
    recent = context[-6000:] if len(context) > 6000 else context
    if system_context:
        recent = f"{system_context}\n\n---\n\n{recent}"

    try:
        # ── Branch 1: Anthropic SDK ───────────────────────────────────────────
        if cfg["url"] is None:
            client = anthropic.Anthropic(api_key=key)
            resp = client.messages.create(
                model=cfg["id"],
                max_tokens=800,
                system=system_prompt,
                messages=[{"role": "user", "content": recent}]
            )
            return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens

        # ── Branch 2: OpenAI-compatible REST ──────────────────────────────────
        else:
            resp = requests.post(
                cfg["url"],
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": cfg["id"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": recent}
                    ],
                    "max_tokens": 800,
                }, timeout=30
            )
            data  = resp.json()
            usage = data.get("usage", {})
            return data["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    except Exception as e:
        return f"❌ {str(e)[:120]}", 0, 0


# ========== MAIN APP ==========
def main():
    st.markdown("""<style>
    .badge-ai   {background:#7C3AED;color:white;padding:2px 10px;border-radius:12px;font-size:0.8em;font-weight:bold;}
    .badge-user {background:#DC2626;color:white;padding:2px 10px;border-radius:12px;font-size:0.8em;font-weight:bold;}
    </style>""", unsafe_allow_html=True)

    # Session state
    if "cost"       not in st.session_state: st.session_state.cost       = 0.0
    if "model_name" not in st.session_state: st.session_state.model_name = None

    archive  = Archive(DB_PATH)
    db_stats = archive.stats()

    # ========== MODEL SELECTION (first launch) ==========
    if st.session_state.model_name is None:
        st.title("🏛️ 20centAI")
        st.subheader("Choose your AI partner")

        available = {name: bool(get_key(cfg["env"])) for name, cfg in MODELS.items()}

        # Group models by provider for a clean grid
        providers = {}
        for name, cfg in MODELS.items():
            provider = name.split()[0]
            providers.setdefault(provider, []).append(name)

        for provider, model_list in providers.items():
            st.markdown(f"**{provider}**")
            cols = st.columns(len(model_list))
            for i, name in enumerate(model_list):
                has_key = available[name]
                with cols[i]:
                    status = "✅ API key found" if has_key else "❌ API key missing"
                    if st.button(
                        f"{name}\n\n`{MODELS[name]['id']}`\n\n{status}",
                        use_container_width=True,
                        disabled=not has_key,
                        key=f"btn_{name}"
                    ):
                        st.session_state.model_name = name
                        st.rerun()
            st.write("")

        if not any(available.values()):
            st.error("No API keys found. Create a `.env` file:")
            st.code("""ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=sk-...
PERPLEXITY_API_KEY=pplx-...
DASHSCOPE_API_KEY=sk-...
GROQ_API_KEY=gsk_...""")
        return

    model_name = st.session_state.model_name

    st.title(f"🏛️ 20centAI · {model_name}")
    st.caption(f"Model: {MODELS[model_name]['id']} · 20-Message-Compression · FTS5 Archive")

    chat_col, side_col = st.columns([3, 1])

    # Load chat once, reuse everywhere in this render cycle
    chat = load_chat()
    n    = count_messages(chat)

    # ========== SIDEBAR ==========
    with side_col:
        st.subheader("⚙️ Status")
        key_ok = bool(get_key(MODELS[model_name]["env"]))
        st.caption(f"{'✅' if key_ok else '❌'} {model_name}")

        if st.button("🔄 Switch model"):
            st.session_state.model_name = None
            st.session_state.cost = 0.0
            st.rerun()

        st.divider()

        st.subheader("🗃️ Archive")
        st.metric("Messages", db_stats["total"])
        st.metric("Topics",   db_stats["topics"])

        st.divider()

        st.subheader("💰 Cost")
        st.metric("Session", f"${st.session_state.cost:.4f}")
        st.caption(f"📊 {n} messages")

        st.divider()

        if n >= COMPRESS_THRESHOLD:
            st.warning(f"⚠️ {n} messages")
            if st.button("🗜️ Compress now"):
                with st.spinner("Compressing..."):
                    chat = compress_chat(chat, model_name)
                    save_chat(chat)
                st.success("✅ Done")
                st.rerun()

        st.divider()

        st.subheader("📦 New topic")
        topic_input = st.text_input("Topic name:", placeholder="e.g. Plugin Architecture")
        if st.button("📁 Archive & restart"):
            if topic_input.strip():
                with st.spinner("Archiving..."):
                    result = archive_and_reset(topic_input.strip(), archive, model_name)
                st.session_state.cost = 0.0
                st.success(result)
                st.rerun()
            else:
                st.warning("Please enter a topic name.")

        st.divider()

        st.subheader("🔍 Search archive")
        search_term = st.text_input("Search:", placeholder="keyword...")
        if search_term and st.button("Search"):
            hits = archive.search(search_term, limit=5)
            st.markdown(hits if hits else "*(no results)*")

        topics = archive.list_topics()
        if topics:
            with st.expander(f"📋 Topics ({len(topics)})"):
                for topic, count, last in topics:
                    st.caption(f"📦 {topic} · {count} msgs · {last[:10]}")

        st.divider()
        st.caption("Tip: type `@archive <keyword>` to inject past context into your message.")

    # ========== CHAT VIEW ==========
    with chat_col:
        # Render each message individually — the role IS the model name, so no confusion
        # when switching models: old messages show who actually wrote them
        model_pattern = "|".join(re.escape(m) for m in ALL_MODEL_NAMES)
        msg_pattern = re.compile(
            rf'\[(\d{{2}}:\d{{2}}) \| (USER|{model_pattern})\]: (.*?)(?=\n\[\d{{2}}:\d{{2}} \| |\Z)',
            re.DOTALL
        )

        # Render header (everything before first message)
        header_match = re.match(r'(.*?)(?=\[\d{2}:\d{2} \|)', chat, re.DOTALL)
        if header_match:
            st.markdown(header_match.group(1))

        for match in msg_pattern.finditer(chat):
            ts, role, body = match.group(1), match.group(2), match.group(3).strip()
            if role == "USER":
                st.markdown(
                    f'<span class="badge-user">👤 YOU {ts}</span>',
                    unsafe_allow_html=True
                )
            else:
                # role IS the model name — always correct, even after switching
                st.markdown(
                    f'<span class="badge-ai">🤖 {role} {ts}</span>',
                    unsafe_allow_html=True
                )
            st.markdown(body)

        st.divider()

        # Input form
        with st.form("chat_form", clear_on_submit=True):
            message = st.text_area(
                "Your message:",
                height=80,
                placeholder="@archive <keyword>  ·  or just write..."
            )
            submitted = st.form_submit_button("🚀 Send", use_container_width=True)

        if submitted and message.strip():
            chat = append_message(chat, "USER", message.strip(), escape_html=True)
            archive.save_message("USER", message.strip())

            # @archiv: extract context for prompt injection — do NOT write it to chat file
            archive_ctx = ""
            if "@archive" in message.lower():
                query = re.sub(r'@archive', '', message, flags=re.IGNORECASE).strip()
                archive_ctx = archive.search(query, limit=3)
                if archive_ctx:
                    st.info(f"📚 Archive context injected for: '{query}'")

            save_chat(chat)

            with st.spinner(f"💭 {model_name} is thinking..."):
                reply, in_tok, out_tok = ai_response(chat, model_name, archive_ctx)

            st.session_state.cost += calc_cost(model_name, in_tok, out_tok)

            # Write model name as role — not generic "AI"
            chat = append_message(chat, model_name, reply)
            archive.save_message(model_name.upper(), reply)
            save_chat(chat)

            # Auto-Komprimierung
            if count_messages(chat) >= COMPRESS_THRESHOLD:
                with st.spinner("🗜️ Auto-compressing..."):
                    chat = compress_chat(chat, model_name)
                    save_chat(chat)

            st.rerun()


if __name__ == "__main__":
    main()
