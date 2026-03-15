# 20centAI 🤖💸

> *"Built the day DeepSeek went down — and I had no Plan B."*

**Switch AI providers with one click. Never lose context when one goes offline.**

---

## The Story

It was a normal workday. DeepSeek open, ready to go — then: **down**.
Completely gone. No fallback, no Plan B.

I had to manually copy everything into another interface. Context lost. Time lost. Flow lost.

One week later, 20centAI existed.

---

## What Is 20centAI?

A minimalist Streamlit chat interface that:

- connects **8 AI providers** — Claude, GPT-4o, Mistral, DeepSeek, Perplexity, Qwen, Groq and more
- **auto-compresses** conversation history — you pay 90% fewer tokens
- runs in a **single Python file** (~600 lines)
- needs **no frameworks** (no LangChain, no vector database)

---

## The Real Problem

Every message you send ships the **entire conversation history** along with it.
After 50+ messages you're paying for tokens you don't need — and most chat tools do nothing about it.

| | Messages | Est. size | Est. cost |
|---|---|---|---|
| Without compression | 1,000 | ~2.5 MB | ~$25.00 |
| With 20centAI | 20 + summary | ~45 KB | **~$2.50** |

**90% saved. Automatically. Invisibly.**

---

## How It Works

```
Messages 1–20  →  compressed into a single summary (~$0.001)
Messages 21–40 →  kept verbatim  ← the AI always sees fresh context
```

When the conversation hits 40 messages, the oldest 20 are summarized automatically
by whichever model you're currently using. You keep full context. You pay 90% less.

---

Existing solutions are either buried inside large frameworks with significant 
setup overhead, or require separate ML models that are overkill for casual API users.

20centAI fills the gap: a **single-file, framework-free** implementation of
rolling-window compression that anyone can read, fork, and modify in an afternoon.

---

## Quickstart

Tested on **Linux** (Ubuntu 22.04+). Should also work on macOS and Windows WSL2.

```bash
# 1. Install dependencies
pip install streamlit anthropic requests python-dotenv

# 2. Create .env file with your API keys
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=sk-...
PERPLEXITY_API_KEY=pplx-...
DASHSCOPE_API_KEY=sk-...
GROQ_API_KEY=gsk_...
EOF

# 3. Run
streamlit run ai_chat_en.py
```

All keys are optional — unavailable providers are disabled automatically at startup.
Add only the keys for the providers you actually use.

---

## Features

| Feature | Details |
|---------|---------|
| 🔄 **One-click provider switch** | Choose from 14 models across 6 providers at startup |
| 🗜️ **Auto-compression** | Triggers at 40 messages, keeps last 20 verbatim |
| 🔍 **Full-text archive search** | SQLite FTS5 index across all past sessions |
| 💰 **Live cost tracking** | Per-session cost visible in the sidebar |
| 📦 **Topic archive** | Save sessions by topic, search with `@archive` |

---

## Supported Models

| Provider | Models | API Key |
|----------|--------|---------|
| Anthropic | Claude Sonnet 4.5, Claude Haiku 4.5 | `ANTHROPIC_API_KEY` |
| OpenAI | GPT-4o, GPT-4o mini | `OPENAI_API_KEY` |
| Mistral | Mistral Large, Mistral Small | `MISTRAL_API_KEY` |
| DeepSeek | DeepSeek V3, DeepSeek R1 | `DEEPSEEK_API_KEY` |
| Perplexity | Sonar Pro | `PERPLEXITY_API_KEY` |
| Qwen (Alibaba) | Qwen Max, Qwen Plus, Qwen Turbo | `DASHSCOPE_API_KEY` |
| Groq | Llama 3.3 70B, Llama 3.1 8B | `GROQ_API_KEY` |

Adding a new OpenAI-compatible provider = **4 lines in the MODELS dict, zero code changes**.

---

## Commands

| Command | Action |
|---------|--------|
| `@archive <keyword>` | Search past sessions, inject matching context into prompt |
| *(anything else)* | Normal chat message |

---

## Configuration

```python
COMPRESS_THRESHOLD = 40  # trigger compression after N messages
KEEP_ORIGINAL      = 20  # always keep last N messages verbatim
```

Adjust to your workflow:
- Long research sessions → `60 / 30`
- Quick daily driver → `30 / 15`

---

## How Compression Works

```
Before (40 messages, ~8000 tokens sent per request):
  [msg 1] ... [msg 40]  ← full history every time

After compression (~1800 tokens):
  [📋 SUMMARY of msg 1–20]  ← ~300 tokens, generated once
  [msg 21] ... [msg 40]     ← last 20 verbatim
```

The summary is generated **once** by the active model, then reused in every
subsequent request — paying ~$0.001 instead of ~$0.10 per round.

---

## Files

```
ai_chat_en.py       ← the entire app (English UI)
ai_chat_de.py       ← same app (German UI)
council_chat.md     ← active chat session (plain text, human-readable)
ai_council.db       ← SQLite archive with FTS5 index
.env                ← API keys
```

---

## Architecture

```
User Input
    │
    ├── @archive?  →  FTS5 search  →  inject context into prompt
    └── normal     →  ai_response()
                           │
                    Two-branch design:
                    ├── Claude       → Anthropic SDK
                    └── Everyone else → OpenAI-compatible REST
                           │
                    save to SQLite (append-only, immutable)
                           │
                    count_messages() >= COMPRESS_THRESHOLD?
                           │
                    compress_chat()  ←  active model summarizes oldest N
---
---

## 🤯 Why It Works
*Your brain compresses 20 years → 2-minute story.
20centAI compresses 1000 messages → 300 tokens.*

**Bio-inspired token compression.**

---

## License
MIT — fork it, ship it, make it yours.

---
*The code was created by [claude.ai](https://claude.ai) in cooperation with [perplexity.ai](https://perplexity.ai) and [deepseek.com](https://deepseek.com).*
