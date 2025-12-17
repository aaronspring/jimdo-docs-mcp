# AGENTS

Purpose: ingest Jimdo Help Center pages, generate dense+sparse embeddings, and expose the Qdrant collection to MCP-capable IDE agents (Codex, GitHub Copilot, Claude Code, VS Code MCP).

## Repo layout (mental model)
- `scrape_jimdo_docs.py` — sitemap-based crawler (LangChain SitemapLoader) that pulls DE help pages, enriches metadata (breadcrumbs, lastmod), and writes `jimdo_docs.csv`.
- `generate_and_upload_embeddings.py` — pipeline that reads the CSV, optionally chunks text, computes FastEmbed dense vectors plus BM25/BM42 sparse vectors, creates/refreshes a Qdrant collection, and upserts points with metadata.
- `main.py` — placeholder entry point (currently just prints a greeting).
- `pyproject.toml` / `uv.lock` — Python deps for uv; includes LangChain, FastEmbed, qdrant-client, pandas, bs4.
- `jimdo_docs.csv` — intermediate dataset produced by the scraper; source for embedding upload.

## Data flow
1) Crawl: `uv run python scrape_jimdo_docs.py` → `jimdo_docs.csv` with `url`, `page_content`, `metadata`.
2) Embed & upload: `uv run python generate_and_upload_embeddings.py --recreate --document-splitting recursive --chunk-size 750 --chunk-overlap 100` → Qdrant collection `jimdo_docs_recursive-750-100` (or variants per args).
3) Serve via MCP: External MCP server (`mcp-server-qdrant` on Aaron’s branch) reads Qdrant using env vars (`QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME`, embedding model names) defined in IDE configs (see README).

## Agent-friendly run snippets
- Generic pattern: `uv run python <script>.py`
- Crawl quick test: `uv run python scrape_jimdo_docs.py`
- List existing Qdrant collections: `uv run python generate_and_upload_embeddings.py --list-collections`
- Upload a test subset: `uv run python generate_and_upload_embeddings.py --n-documents 20 --document-splitting tokens --chunk-size 500 --chunk-overlap 50 --recreate`

## Config & environment
- Loads secrets from `.envrc` (via `python-dotenv`): must set `QDRANT_URL`, `QDRANT_API_KEY`; optional collection/model overrides.
- Embedding models: dense `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (default) and `jina-embeddings-v2-base-de`; sparse `Qdrant/bm25` and `Qdrant/bm42-all-minilm-l6-v2-attentions`.
- Collection naming: base `jimdo_docs`; suffixes encode splitter and params (e.g., `_recursive-750-100`, add `_test` when `--n-documents` is used).

## Notes for IDE agents
- Treat `jimdo_docs.csv` as generated; don’t hand-edit.
- MCP config examples for Codex/VS Code/Claude are in `README.md`; keep server args/env in sync if you modify defaults here.
- Network calls (Qdrant, sitemap) require env vars and outbound access.
- If adding tools/scripts, mirror their usage here so agents surface correct commands.
