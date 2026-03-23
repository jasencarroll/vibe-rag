# Memory Event Convention v1

This is the small shared convention used by write adapters that turn external events into durable `vibe-rag` memory.

## Core Fields

Adapters using this convention write these metadata keys:

- `convention = "memory_event_v1"`
- `event_at` as an ISO 8601 UTC timestamp
- `thread = { id, title }`
- `adapter` with the adapter name

Rules:

- `thread.id` should be stable for the life of the stream.
- `thread.title` should be human-readable and can change if you need a clearer label later.
- `event_at` should represent when the event happened, not when retrieval happened.
- Adapters can add more metadata, but they should not reuse these keys with different meanings.

## Daily Note Adapter

`ingest_daily_note` uses:

- `adapter = "daily_note"`
- `capture_kind = "adapter_daily_note"`
- `note_date = YYYY-MM-DD`
- default `thread.id = "daily:YYYY-MM-DD"`
- default `thread.title = "Daily Note YYYY-MM-DD"`

If `event_at` is omitted, it defaults to `YYYY-MM-DDT00:00:00Z`.

## PR Outcome Adapter

`ingest_pr_outcome` uses:

- `adapter = "pr_outcome"`
- `capture_kind = "adapter_pr_outcome"`
- `pr_number`
- `pr_title`
- `outcome`
- optional `issue_id`, `branch`, `commit_sha`, `pr_url`
- default `thread.id = "pr:<number>"`
- default `thread.title = "PR #<number>: <title>"`

If `event_at` is omitted, it defaults to the current UTC timestamp at ingestion time.
