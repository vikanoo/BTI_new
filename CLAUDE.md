# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Telegram bot that analyzes apartment floor plans (BTI documents) and detects unauthorized renovations (перепланировка). The bot compares official BTI plans with user-submitted photos to classify changes under Russian housing regulations.

**Current status:** Week 1 MVP development (started March 16, 2026). Core workflow skeleton built; AI model integrations (Gemini, GPT-4o) not yet called in production.

## Technology Stack

This is a **no-code/low-code project** — there is no traditional build system, compiled code, or test runner.

| Layer | Technology |
|-------|-----------|
| Workflow orchestration | N8N (self-hosted on VPS via Docker Compose) |
| User interface | Telegram Bot API |
| Database + storage | Supabase (PostgreSQL + pgvector, Storage bucket `bti-files`) |
| Vision analysis | Google Gemini 2.0 Flash |
| Final analysis + RAG | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Image annotation (planned) | Python + Flask + PIL/Pillow |

**All logic lives in the N8N workflow** exported at [Workflows/BTI_NEW.json](Workflows/BTI_NEW.json).

## Architecture

### Request Flow

```
User (Telegram) → N8N Webhook (POST /webhook/bti-bot)
  → Whitelist check (Supabase: allowed_users table)
  → Switch by message type (document / photo / text / callback_query)
    → Read session step from Supabase (sessions table)
    → Route by step → Execute action
  → Update session → Reply via Telegram
```

### Dialog State Machine

Session state is stored in the `sessions` table (Supabase) under a `step` field:

| Step | Description |
|------|-------------|
| `idle` | Initial / no active session |
| `awaiting_plan` | Waiting for user to upload floor plan |
| `plan_received` | Plan stored, Gemini analysis pending |
| `collecting_photos` | Room photo collection loop active |
| `photos_complete` | All photos received, GPT-4o analysis pending |

Photo collection progress is tracked via `shots_status` (JSONB column) with per-shot states: `pending` → `requested` → `received`.

### Message Routing Pattern

Two-level switch in the N8N workflow:
1. **By type**: `document`, `photo`, `text`, `callback_query`
2. **By step**: current session state (read from Supabase before routing)

### RAG Architecture (planned, Week 2)

Three PDF documents in [БазаЗнаний/](БазаЗнаний/) contain Russian housing norms. These will be chunked, embedded via `text-embedding-3-small`, stored in the `embeddings` table (pgvector), and queried at analysis time to provide GPT-4o with relevant regulatory context.

## Key Files

- [Workflows/BTI_NEW.json](Workflows/BTI_NEW.json) — N8N workflow export (the entire bot logic)
- [REQUIREMENTS.md](REQUIREMENTS.md) — Full system specification: user flows, API details, cost model, Supabase schema, limitations
- [PLAN_WEEK_1.md](PLAN_WEEK_1.md) — Day-by-day development plan for Week 1

## Supabase Schema

```sql
-- Access control
allowed_users (chat_id BIGINT PRIMARY KEY)

-- Dialog state
sessions (
  chat_id        BIGINT PRIMARY KEY,
  step           TEXT,
  plan_url       TEXT,
  shots_status   JSONB,  -- [{id, description, status}]
  created_at     TIMESTAMPTZ,
  updated_at     TIMESTAMPTZ
)

-- RAG knowledge base (Phase 2)
embeddings (
  id        UUID PRIMARY KEY,
  content   TEXT,
  embedding VECTOR(1536),  -- pgvector, text-embedding-3-small
  metadata  JSONB
)
```

## Development Workflow

Since there is no build system, "development" means:
1. Edit workflow logic in the N8N UI or by editing `Workflows/BTI_NEW.json` directly
2. Import updated JSON into N8N via the workflow import feature
3. Test via Telegram (live bot) — there is no local test runner

**To deploy workflow changes:** Import `BTI_NEW.json` into the N8N instance via Settings → Import Workflow.

## Important Constraints

- **Whitelist-only access**: Only `chat_id`s in `allowed_users` can interact with the bot; all others are silently ignored.
- **Cost target**: ~$0.28–0.40 per full analysis. Dialog state management is handled by N8N (no AI), reducing unnecessary API calls.
- **Language**: All user-facing messages are in Russian.
- **Scope limitation**: The bot detects obvious structural changes (wall removal, room merging) only — not fine details. Results are informational, not legally binding.

# Project Rules

## 1. NEVER restore previous versions without confirmation
ALWAYS ask before restoring any backup or previous version

## 2. Security - NEVER commit credentials
- config.json contains API key - NEVER commit
- .env files - NEVER commit
- ALWAYS check git status before commits

## 3. Git workflow
- Main branch: main
- ALWAYS create descriptive commit messages

## 4. Before deleting
ALWAYS ask before deleting any file

## 5. Project structure
- Workflows/ - workflows
- KnowledgeBase/ - documentation
- See: Workflows/REQUIREMENTS.md for details