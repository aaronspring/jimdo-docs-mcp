Jimdo docs embeddings and Qdrant MCP usage.

## Prerequisites
- `uv` available (project depends on Python 3.12+).
- option to upload embeddings: `jimdo_docs.csv` generated via `uv run python scrape_jimdo_docs.py` - requires QDRANT_API_KEY

## Generate embeddings and upload to Qdrant
```bash
cd jimdo-docs-mcp
uv run python generate_and_upload_embeddings.py --recreate \
  --document-splitting recursive \
  --chunk-size 750 --chunk-overlap 100
```
- Collections are named `jimdo_docs` plus the splitting suffix (e.g., `jimdo_docs_recursive-750-100`).
- Use `--list-collections` to inspect existing Qdrant collections.
- Use `--n-documents` to run a quick test subset.

## Using the Qdrant collection via the Qdrant MCP server
These steps assume you have the Qdrant MCP server binary available. For hybrid dense+sparse search, use the fork from Aaron Spring’s branch:
- Install via uvx: `uvx --from git+https://github.com/aaronspring/mcp-server-qdrant@feat/configurable-sparse-embedding mcp-server-qdrant`

One-click install banner for VS Code MCP (prompts for your Qdrant URL/API key):
[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_Jimdo_Qdrant_MCP-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](vscode:mcp/install?name=jimdo-docs-qdrant&config=%7B%22type%22%3A%22stdio%22%2C%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22--from%22%2C%22git%2Bhttps%3A//github.com/aaronspring/mcp-server-qdrant%40feat/configurable-sparse-embedding%22%2C%22mcp-server-qdrant%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3Aqdrant-url%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3Aqdrant-api-key%7D%22%2C%22COLLECTION_NAME%22%3A%22jimdo_docs_recursive-750-100%22%2C%22EMBEDDING_MODEL%22%3A%22sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2%22%2C%22SPARSE_EMBEDDING_NAME%22%3A%22fast-bm42-all-minilm-l6-v2-attentions%22%7D%7D)

1) Start the Qdrant MCP server (replace values accordingly):
```bash
qdrant-mcp --url https://95d213c9-ed91-4fbd-8d89-3aa4f0b972f4.eu-central-1-0.aws.cloud.qdrant.io --api-key eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJyIn0.DoWUjYic92IDLsYevtk9oFCHSqooOJ0tcpAntcQVbT4 \
  --collection jimdo_docs_recursive-750-100
```

2) Configure Claude Code / VS Code to point at the MCP server.
- Create or edit your Claude Code config (e.g., `~/.claude/code/config.json`):
```json
{
  "mcpServers": {
    "qdrant": {
      "command": "uvx",
        "args": [
          "--from",
          "git+https://github.com/aaronspring/mcp-server-qdrant@feat/configurable-sparse-embedding",
          "mcp-server-qdrant"
        ],
        "env": {
        "QDRANT_URL": "https://95d213c9-ed91-4fbd-8d89-3aa4f0b972f4.eu-central-1-0.aws.cloud.qdrant.io",
        "QDRANT_API_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJyIn0.DoWUjYic92IDLsYevtk9oFCHSqooOJ0tcpAntcQVbT4",
        "COLLECTION_NAME": "jimdo_docs_recursive-750-100",
        "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "SPARSE_EMBEDDING_NAME": "fast-bm42-all-minilm-l6-v2-attentions",
        "TOOL_FIND_DESCRIPTION": "Search Green Party Hamburg documents using dense semantic search. Use natural language queries (e.g., 'Verkehrspolitik und Mobilitätswende', 'bezahlbarer Wohnraum Hamburg') rather than just keywords. Returns up to 10 most relevant document excerpts. Always include document URLs in your answer to allow users to verify sources. Keep queries to qdrant very similar to the user input. Don't add too many keywords.",
        "TOOL_HYBRID_FIND_DESCRIPTION": "Advanced configurable hybrid search for Green Party Hamburg documents. Use natural language queries (e.g., 'Klimaschutz Maßnahmen und erneuerbare Energien') to leverage both semantic understanding and keyword matching. ALWAYS set these parameters: dense_limit=10 (semantic results), sparse_limit=10 (keyword results), final_limit=10 (merged results). Use fusion_method='dbsf'. Include document URLs in your answer. Keep queries to qdrant very similar to the user input. Don't add too many keywords.",
        "TOOL_STORE_DESCRIPTION": "NEVER USE"
     }   
    }
  }
}
```
- Reload the Claude extension; the Qdrant MCP server will expose tools for similarity search against the Jimdo collection.

3) Query from Claude Code
- Ask Claude to “use the qdrant MCP server to search for Jimdo docs about X”.
- The server returns the matching chunks you uploaded; include them in prompts or code references as needed.

## Notes
- Default dense models: `paraphrase-multilingual-MiniLM-L12-v2` and `jina-embeddings-v2-base-de`.
- Sparse models: `Qdrant/bm25` and `Qdrant/bm42-all-minilm-l6-v2-attentions`.
- Collections include both dense and sparse vectors for hybrid retrieval.
